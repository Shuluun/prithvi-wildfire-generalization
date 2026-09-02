"""M2: spectral Random Forest baseline on the frozen M1.6 protocol.

Runs 5-fold event-disjoint CV over the 576-event primary population
(matched MTBS Wildfire events) using the frozen outer split and the frozen
inner train/val split (scripts/m2_make_inner_split.py). One out-of-fold
prediction per primary event.

Protocols (--protocol):
  event     PRIMARY: split_k5_event_seed42.csv
  spatial   spatial-overlap-grouped sensitivity (zero train/test footprint
            overlap); split_k5_spatial_seed42.csv
  tiledisjoint  stricter sensitivity (deferred until event+spatial are sane)

Pixel-sampling policy (frozen): uniform sample up to CAP valid pixels per
training chip, without replacement, natural class balance preserved,
deterministic from (global seed + image_id). Validation/test are NOT
subsampled — all valid pixels are evaluated; -1 (no-data) is always excluded.

RF config (frozen): n_estimators=300, random_state=42, n_jobs=-1,
class_weight="balanced_subsample", max_features="sqrt". Threshold selected on
inner validation by maximizing pooled burned Dice over a fixed grid.

Usage (from repo root):
  python scripts/m2_spectral_baseline.py --protocol event
  python scripts/m2_spectral_baseline.py --protocol spatial
  python scripts/m2_spectral_baseline.py --smoke   # NON-SCIENTIFIC dry run
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.data.splits import SPLITS_DIR  # noqa: E402
from src.evaluation.bootstrap import summarize_events  # noqa: E402
from src.evaluation.metrics import per_chip_metrics, select_threshold  # noqa: E402
from src.features.spectral import (  # noqa: E402
    FEATURE_NAMES, chip_rng, load_features, sample_valid_indices,
    valid_features_labels,
)

SEED = 42
CAP_DEFAULT = 2048
N_ESTIMATORS_DEFAULT = 300
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
METRIC_COLS = ["iou", "dice", "precision", "recall", "auprc", "auroc"]

RF_KWARGS = dict(n_estimators=N_ESTIMATORS_DEFAULT, random_state=SEED,
                 n_jobs=-1, class_weight="balanced_subsample",
                 max_features="sqrt")


def load_primary():
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    assert primary["event_id"].is_unique and primary["image_id"].is_unique
    return primary


def load_inner(protocol):
    return pd.read_csv(os.path.join(
        SPLITS_DIR, f"split_k5_{protocol}_inner_seed42.csv"))


def merged_path_of(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir, image_id + "_merged.tif")


def train_fold(train_rows, img_split, cap, n_estimators=N_ESTIMATORS_DEFAULT):
    """Fit the RF on sampled valid pixels of the inner-train chips.

    Returns (clf, sampling_audit_rows). Sampling is uniform (natural class
    balance) and deterministic per chip.
    """
    Xs, ys = [], []
    audit = []
    for image_id, event_id in zip(train_rows["image_id"], train_rows["event_id"]):
        mp = merged_path_of(image_id, img_split[image_id])
        x, mask = load_features(mp)
        rng = chip_rng(SEED, image_id)
        idx = sample_valid_indices(mask, cap, rng)
        n_valid = int((mask >= 0).sum())
        n_burned = int((mask == 1).sum())
        y = mask[idx].astype(np.int64)
        Xs.append(x[idx])
        ys.append(y)
        audit.append({
            "image_id": image_id, "event_id": event_id,
            "n_valid": n_valid, "n_burned_avail": n_burned,
            "n_sampled": int(len(idx)),
            "n_sampled_burned": int((y == 1).sum()),
            "sampled_burn_fraction": float((y == 1).mean()),
        })
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    clf = RandomForestClassifier(**dict(RF_KWARGS, n_estimators=n_estimators))
    clf.fit(X, y)
    return clf, audit


def threshold_on_val(clf, val_rows, img_split):
    """Pool inner-val valid pixels, select the burned-Dice-maximizing
    threshold. Returns (threshold, dice_at_threshold)."""
    y_true_parts, y_prob_parts = [], []
    for image_id in val_rows["image_id"]:
        mp = merged_path_of(image_id, img_split[image_id])
        x, mask = load_features(mp)
        xv, yv = valid_features_labels(x, mask)
        if len(xv) == 0:
            continue
        proba = clf.predict_proba(xv)[:, 1].astype(np.float32)
        y_true_parts.append(yv.astype(np.int8))
        y_prob_parts.append(proba)
    y_true = np.concatenate(y_true_parts)
    y_prob = np.concatenate(y_prob_parts)
    thr, dice = select_threshold(y_true, y_prob, THRESHOLD_GRID)
    return thr, dice


def predict_test(clf, test_rows, img_split, threshold, out_pred_dir):
    """Evaluate all valid pixels of every outer-test chip at the frozen
    threshold. Returns one OOF metrics row per event; saves uint8 binary
    prediction maps for figure use."""
    oof_rows = []
    for image_id, event_id in zip(test_rows["image_id"], test_rows["event_id"]):
        mp = merged_path_of(image_id, img_split[image_id])
        x, mask = load_features(mp)
        xv, yv = valid_features_labels(x, mask)
        proba = clf.predict_proba(xv)[:, 1].astype(np.float32)
        m = per_chip_metrics(yv, proba, threshold)
        oof_rows.append({"event_id": event_id, "image_id": image_id, **m})
        if out_pred_dir is not None:
            pred = (proba >= threshold).astype(np.uint8)
            np.save(os.path.join(out_pred_dir, f"{event_id}.npy"), pred)
    return oof_rows


def _protocol_name(protocol):
    return {"event": "PRIMARY event-disjoint K5",
            "spatial": "spatial-overlap-grouped K5",
            "tiledisjoint": "tile-disjoint K5"}[protocol]


def run_protocol(protocol, cap=CAP_DEFAULT, n_estimators=N_ESTIMATORS_DEFAULT):
    primary = load_primary()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = load_inner(protocol)
    # same primary events exactly
    assert set(inner["event_id"]) == set(primary["event_id"])

    out_dir = os.path.join(RESULTS_ROOT, "m2", protocol)
    pred_dir = os.path.join(out_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    rf_kwargs = dict(RF_KWARGS, n_estimators=n_estimators)
    oof_all, audit_all, fold_rows, thr_rows = [], [], [], []
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        train_rows = grp[grp["role"] == "train"]
        val_rows = grp[grp["role"] == "val"]
        test_rows = grp[grp["role"] == "test"]

        clf, audit = train_fold(train_rows, img_split, cap,
                                n_estimators=n_estimators)
        for a in audit:
            a["fold"] = f
        audit_all.extend(audit)

        thr, dice_val = threshold_on_val(clf, val_rows, img_split)
        oof = predict_test(clf, test_rows, img_split, thr, pred_dir)
        for r in oof:
            r["fold"] = f
            r["threshold"] = thr
        oof_all.extend(oof)

        fold_rows.append({
            "fold": f, "n_inner_train": int(len(train_rows)),
            "n_inner_val": int(len(val_rows)),
            "n_outer_test": int(len(test_rows)),
            "threshold": thr, "inner_val_dice": dice_val,
        })
        thr_rows.append({"fold": f, "threshold": thr,
                         "inner_val_dice_at_threshold": dice_val})
        print(f"[{protocol}] fold {f}: train={len(train_rows)} "
              f"val={len(val_rows)} test={len(test_rows)} "
              f"threshold={thr:.4f} inner_val_dice={dice_val:.4f}")

    oof = pd.DataFrame(oof_all)
    audit = pd.DataFrame(audit_all)
    folds = pd.DataFrame(fold_rows)
    thrs = pd.DataFrame(thr_rows)

    # exactly one OOF prediction per primary event
    assert oof["event_id"].is_unique, "duplicate OOF prediction for an event"
    assert len(oof) == len(primary), "missing OOF prediction for an event"

    col_order = ["event_id", "image_id", "fold", "threshold",
                 "iou", "dice", "precision", "recall", "auprc", "auroc",
                 "true_burn_fraction", "pred_burn_fraction", "n_valid"]
    oof = oof[col_order].sort_values("event_id").reset_index(drop=True)
    oof.to_csv(os.path.join(out_dir, "m2_event_oof_predictions.csv"),
               index=False)
    folds.to_csv(os.path.join(out_dir, "m2_fold_summary.csv"), index=False)
    thrs.to_csv(os.path.join(out_dir, "m2_thresholds.csv"), index=False)
    audit.to_csv(os.path.join(out_dir, "m2_sampling_audit.csv"), index=False)
    summary = summarize_events(oof, METRIC_COLS)
    summary.to_csv(os.path.join(out_dir, "m2_metrics_summary.csv"), index=False)

    with open(os.path.join(out_dir, "m2_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "protocol": protocol, "protocol_name": _protocol_name(protocol),
            "seed": SEED, "cap_pixels_per_train_chip": cap,
            "rf_kwargs": rf_kwargs, "threshold_grid": THRESHOLD_GRID.tolist(),
            "n_primary_events": int(len(primary)),
            "metric_columns": METRIC_COLS,
        }, fh, indent=2)

    print(f"\n[{protocol}] OOF predictions: {len(oof)} events")
    print(summary[["metric", "n_events", "mean", "median", "bootstrap_ci_lo",
                   "bootstrap_ci_hi"]].to_string(index=False))
    print(f"outputs -> {out_dir}")
    return oof, summary


def smoke(cap=CAP_DEFAULT):
    """NON-SCIENTIFIC dry run on a tiny subset (fold 1 only)."""
    primary = load_primary()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = load_inner("event")
    grp = inner[inner["outer_fold"] == 1]
    train_rows = grp[grp["role"] == "train"].head(6)
    val_rows = grp[grp["role"] == "val"].head(4)
    test_rows = grp[grp["role"] == "test"].head(4)

    # channel-order + index-range diagnostics on one chip
    mp0 = merged_path_of(train_rows["image_id"].iloc[0],
                         img_split[train_rows["image_id"].iloc[0]])
    x0, mask0 = load_features(mp0)
    ndvi = x0[:, 6][mask0 >= 0]
    nbr = x0[:, 7][mask0 >= 0]
    n_nodata = int((mask0 == -1).sum())
    print("== NON-SCIENTIFIC SMOKE TEST ==")
    print(f"feature names: {FEATURE_NAMES} (dim={len(FEATURE_NAMES)})")
    print(f"one chip: X.shape={x0.shape}, mask.shape={mask0.shape}")
    print(f"nodata(-1) pixels in chip: {n_nodata}")
    print(f"NDVI range (valid): [{ndvi.min():.4f}, {ndvi.max():.4f}]")
    print(f"NBR  range (valid): [{nbr.min():.4f}, {nbr.max():.4f}]")
    assert x0.shape[1] == 8
    assert np.isfinite(ndvi).all() and np.isfinite(nbr).all()

    # determinism of pixel sampling
    rng_a = chip_rng(SEED, train_rows["image_id"].iloc[0])
    rng_b = chip_rng(SEED, train_rows["image_id"].iloc[0])
    assert np.array_equal(sample_valid_indices(mask0, cap, rng_a),
                          sample_valid_indices(mask0, cap, rng_b))
    print("deterministic pixel sampling: OK")

    clf, audit = train_fold(train_rows, img_split, cap, n_estimators=20)
    assert audit and all(a["n_sampled"] > 0 for a in audit)
    assert not any(a["n_sampled_burned"] < 0 for a in audit)
    thr, dice = threshold_on_val(clf, val_rows, img_split)
    print(f"RF fitted (n_estimators={clf.n_estimators}); "
          f"threshold={thr:.4f}, inner_val_dice={dice:.4f}")
    oof = predict_test(clf, test_rows, img_split, thr, None)
    for r in oof:
        print(f"  test {r['event_id'][:24]}... iou={r['iou']:.4f} "
              f"dice={r['dice']:.4f} auprc={r['auprc']:.4f} "
              f"auroc={r['auroc']:.4f} true_burn={r['true_burn_fraction']:.4f}")
    out = os.path.join(RESULTS_ROOT, "m2", "smoke")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(oof).to_csv(os.path.join(out, "m2_smoke_oof.csv"), index=False)
    with open(os.path.join(out, "m2_smoke_NONSCIENTIFIC.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("M2 smoke test — NON-SCIENTIFIC IMPLEMENTATION TEST ONLY.\n")
        fh.write(f"feature dim={len(FEATURE_NAMES)}; NDVI range [{ndvi.min():.4f},"
                 f"{ndvi.max():.4f}]; NBR range [{nbr.min():.4f},{nbr.max():.4f}];"
                 f" nodata={n_nodata}; threshold={thr:.4f}.\n")
    print(f"-> {out}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["event", "spatial", "tiledisjoint"],
                    default="event")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    ap.add_argument("--n-estimators", type=int, default=N_ESTIMATORS_DEFAULT)
    args = ap.parse_args()
    if args.smoke:
        smoke(cap=args.cap)
    else:
        run_protocol(args.protocol, cap=args.cap,
                     n_estimators=args.n_estimators)

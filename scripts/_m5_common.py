"""Shared training/evaluation plumbing for the M5 decoder comparison.

Every M5 model is trained under the IDENTICAL protocol as M2 (spectral RF) and
M4b (linear probe): the frozen 576-event Wildfire population, the frozen
event-disjoint inner K5 split, the deterministic <=2048 valid-pixel supervised
budget per train chip (global seed + image_id), the frozen inner-val threshold
policy (maximize pooled burned Dice over a 99-point grid), and outer test
touched once per fold. This module only factors out the common loop; the
scientific protocol is unchanged.

Usage: each driver script supplies a ``build_model()`` factory (fresh model per
fold) and a ``load_features(image_id, split_dir) -> tensor`` function whose
model forward returns a (1, 1, 512, 512) logit map.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.data.splits import SPLITS_DIR  # noqa: E402
from src.evaluation.bootstrap import bootstrap_ci, summarize_events  # noqa: E402
from src.evaluation.metrics import per_chip_metrics, select_threshold  # noqa: E402
from src.features.spectral import chip_rng, sample_valid_indices  # noqa: E402

SEED = 42
CAP = 2048
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_COLS = ["iou", "dice", "precision", "recall", "auprc", "auroc"]
OPT = dict(lr=1e-3, weight_decay=1e-4, max_epochs=50, patience=5)

CACHE_FEAT = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache", "features")


# --------------------------------------------------------------------------- #
# mask / feature loading
# --------------------------------------------------------------------------- #
def mask_path_of(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                        image_id + "_merged.tif").replace("_merged.tif", ".mask.tif")


def load_mask(image_id, split_dir):
    import rasterio
    with rasterio.open(mask_path_of(image_id, split_dir)) as ds:
        return ds.read(1).astype(np.int8)  # (512, 512)


def load_prithvi_concat(image_id, split_dir=None):
    """(1, 4096, 32, 32) fp32 — concatenated frozen layers [5,11,17,23]."""
    a = np.load(os.path.join(CACHE_FEAT, f"{image_id}.npy"))  # (4,1024,32,32) fp16
    a = a.astype(np.float32).reshape(1, -1, a.shape[2], a.shape[3])
    return torch.from_numpy(a).to(DEVICE)


def load_prithvi_layers(image_id, split_dir=None):
    """(1, 4, 1024, 32, 32) fp32 — frozen layers kept separate (for the decoder)."""
    a = np.load(os.path.join(CACHE_FEAT, f"{image_id}.npy"))  # (4,1024,32,32) fp16
    return torch.from_numpy(a.astype(np.float32))[None].to(DEVICE)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --------------------------------------------------------------------------- #
# per-chip forward helpers (model returns (1,1,512,512) logits)
# --------------------------------------------------------------------------- #
def sampled_pixel_logits_labels(model, load_features, image_id, split_dir, rng,
                                cap=CAP):
    mask = load_mask(image_id, split_dir)
    idx = sample_valid_indices(mask.ravel(), cap, rng)
    rows, cols = idx // mask.shape[1], idx % mask.shape[1]
    y = mask.ravel()[idx].astype(np.int64)
    logits = model(load_features(image_id, split_dir))
    return logits[0, 0, rows, cols], torch.from_numpy(y).to(DEVICE)


def eval_chip_logits(model, load_features, image_id, split_dir):
    mask = load_mask(image_id, split_dir)
    valid = mask >= 0
    logits = model(load_features(image_id, split_dir))[0, 0]  # (512,512)
    yv = mask[valid].astype(np.int64)
    lv = logits[valid].detach().cpu().numpy().astype(np.float32)
    return lv, yv


def fold_val_dice(model, load_features, val_rows, img_split):
    yt, yp = [], []
    for image_id in val_rows["image_id"]:
        lv, yv = eval_chip_logits(model, load_features, image_id,
                                  img_split[image_id])
        if len(yv) == 0:
            continue
        yt.append(yv)
        yp.append(lv)
    if not yt:
        return 0.0, None
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    thr, dice = select_threshold(yt, _sigmoid(yp), THRESHOLD_GRID)
    return dice, thr


def train_model(model, load_features, train_rows, val_rows, img_split,
                pos_weight, opt=OPT):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"],
                                  weight_decay=opt["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],
                                                           device=DEVICE))
    best = {"epoch": -1, "val_dice": -1.0, "state": None, "loss": None}
    patience_left = opt["patience"]

    for epoch in range(1, opt["max_epochs"] + 1):
        model.train()
        losses = []
        for image_id in train_rows["image_id"]:
            rng = chip_rng(SEED, image_id)
            logits_s, y = sampled_pixel_logits_labels(
                model, load_features, image_id, img_split[image_id], rng, CAP)
            if len(y) == 0:
                continue
            loss = loss_fn(logits_s, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")

        # early-stopping monitor: inner-val Dice (dropout/BN off for determinism)
        model.eval()
        val_dice, _ = fold_val_dice(model, load_features, val_rows, img_split)
        if val_dice > best["val_dice"]:
            best = {"epoch": epoch, "val_dice": val_dice,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "loss": mean_loss}
            patience_left = opt["patience"]
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    model.load_state_dict(best["state"])
    model.eval()
    return model, best


# --------------------------------------------------------------------------- #
# full 5-fold experiment runner
# --------------------------------------------------------------------------- #
def run_experiment(build_model, load_features, out_root, tag, arch_desc,
                   extra_config=None):
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    assert set(inner["event_id"]) == set(primary["event_id"])

    os.makedirs(out_root, exist_ok=True)
    pred_dir = os.path.join(out_root, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    n_trainable = None
    oof_all, fold_rows, thr_rows, train_rows_out = [], [], [], []
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        train_rows = grp[grp["role"] == "train"]
        val_rows = grp[grp["role"] == "val"]
        test_rows = grp[grp["role"] == "test"]

        n_pos = n_neg = 0
        for image_id in train_rows["image_id"]:
            m = load_mask(image_id, img_split[image_id])
            n_pos += int((m == 1).sum())
            n_neg += int((m == 0).sum())
        pos_weight = n_neg / n_pos if n_pos else 1.0

        model = build_model()
        if n_trainable is None:
            n_trainable = int(model.n_trainable)
        model, best = train_model(model, load_features, train_rows, val_rows,
                                  img_split, pos_weight)
        val_dice, thr = fold_val_dice(model, load_features, val_rows, img_split)

        for image_id, event_id in zip(test_rows["image_id"],
                                      test_rows["event_id"]):
            lv, yv = eval_chip_logits(model, load_features, image_id,
                                      img_split[image_id])
            yp = _sigmoid(lv)
            m = per_chip_metrics(yv, yp, thr)
            oof_all.append({"event_id": event_id, "image_id": image_id,
                            "fold": f, "threshold": thr, **m})
            np.save(os.path.join(pred_dir, f"{event_id}.npy"),
                    (yp >= thr).astype(np.uint8))

        fold_rows.append({"fold": f, "n_inner_train": int(len(train_rows)),
                          "n_inner_val": int(len(val_rows)),
                          "n_outer_test": int(len(test_rows)),
                          "threshold": thr, "inner_val_dice": val_dice,
                          "best_epoch": best["epoch"],
                          "train_loss": best["loss"]})
        thr_rows.append({"fold": f, "threshold": thr,
                         "inner_val_dice_at_threshold": val_dice})
        train_rows_out.append({"fold": f, "n_trainable": n_trainable,
                               "pos_weight": pos_weight,
                               "best_epoch": best["epoch"],
                               "train_loss": best["loss"]})
        print(f"[{tag}] fold {f}: train={len(train_rows)} val={len(val_rows)} "
              f"test={len(test_rows)} thr={thr:.4f} val_dice={val_dice:.4f} "
              f"best_epoch={best['epoch']} loss={best['loss']:.4f} "
              f"trainable={n_trainable}")

    oof = pd.DataFrame(oof_all)
    assert oof["event_id"].is_unique and len(oof) == len(primary)
    col_order = ["event_id", "image_id", "fold", "threshold",
                 "iou", "dice", "precision", "recall", "auprc", "auroc",
                 "true_burn_fraction", "pred_burn_fraction", "n_valid"]
    oof = oof[col_order].sort_values("event_id").reset_index(drop=True)
    oof.to_csv(os.path.join(out_root, f"{tag}_event_oof_predictions.csv"),
               index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out_root, f"{tag}_fold_summary.csv"),
                                   index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(out_root, f"{tag}_thresholds.csv"),
                                   index=False)
    pd.DataFrame(train_rows_out).to_csv(
        os.path.join(out_root, f"{tag}_training_summary.csv"), index=False)
    summary = summarize_events(oof, METRIC_COLS)
    summary.to_csv(os.path.join(out_root, f"{tag}_metrics_summary.csv"),
                   index=False)

    config = {
        "tag": tag,
        "architecture": arch_desc,
        "trainable_params": n_trainable,
        "seed": SEED,
        "cap_pixels_per_train_chip": CAP,
        "opt": OPT,
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "n_primary_events": int(len(primary)),
        "device": DEVICE,
    }
    if extra_config:
        config.update(extra_config)
    with open(os.path.join(out_root, f"{tag}_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"\n[{tag}] OOF predictions: {len(oof)} events, "
          f"trainable={n_trainable}")
    print(summary[["metric", "n_events", "mean", "median", "bootstrap_ci_lo",
                   "bootstrap_ci_hi"]].to_string(index=False))
    print(f"outputs -> {out_root}")
    return oof, summary


# --------------------------------------------------------------------------- #
# paired event-level delta + bootstrap CI vs a baseline
# --------------------------------------------------------------------------- #
def paired_delta(model_oof, baseline_path, baseline_name, model_name,
                 metric_cols=METRIC_COLS, out_path=None):
    base = pd.read_csv(baseline_path)
    merged = base.merge(model_oof, on="event_id",
                        suffixes=(f"_{baseline_name}", f"_{model_name}"))
    rows = []
    for mc in metric_cols:
        bc, mc2 = f"{mc}_{baseline_name}", f"{mc}_{model_name}"
        d = (merged[mc2] - merged[bc]).to_numpy(float)
        lo, hi = bootstrap_ci(d, seed=42)
        rows.append({"metric": mc,
                     f"{baseline_name}_mean": float(merged[bc].mean()),
                     f"{model_name}_mean": float(merged[mc2].mean()),
                     "delta_mean": float(np.nanmean(d)),
                     "delta_median": float(np.nanmedian(d)),
                     "delta_ci_lo": lo, "delta_ci_hi": hi})
    out = pd.DataFrame(rows)
    if out_path:
        out.to_csv(out_path, index=False)
    return out

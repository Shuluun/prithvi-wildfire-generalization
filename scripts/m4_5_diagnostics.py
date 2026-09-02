"""M4.5 diagnostics (data-only) — error overlap, stratifications, calibration,
and boundary-resolution analysis of the frozen Prithvi linear probe vs the frozen
spectral RF.

No training is performed here: it consumes the already-saved per-event OOF
predictions (results/m2/event, results/m4_linear_probe) and per-event binary maps,
plus data/metadata/event_attributes.csv. All outputs go to results/m4_5/.

Usage (from repo root):
  python scripts/m4_5_diagnostics.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402

OUT = os.path.join(RESULTS_ROOT, "m4_5")
RF_OOF = os.path.join(RESULTS_ROOT, "m2", "event", "m2_event_oof_predictions.csv")
PR_OOF = os.path.join(RESULTS_ROOT, "m4_linear_probe", "m4_event_oof_predictions.csv")
RF_PRED = os.path.join(RESULTS_ROOT, "m2", "event", "preds")
PR_PRED = os.path.join(RESULTS_ROOT, "m4_linear_probe", "preds")

BURN_BINS = [(-1, 0.01, "<0.01"), (0.01, 0.05, "0.01-0.05"),
             (0.05, 0.20, "0.05-0.20"), (0.20, 1.01, ">=0.20")]


def _attrs():
    a = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    return a[a["incid_type"] == "Wildfire"].copy()


def merged():
    rf = pd.read_csv(RF_OOF)
    pr = pd.read_csv(PR_OOF)
    m = rf.merge(pr, on="event_id", suffixes=("_rf", "_pr"))
    a = _attrs()[["event_id", "fire_year", "hls_tile", "burned_fraction"]]
    m = m.merge(a, on="event_id", how="left")
    m["latband"] = m["hls_tile"].str[2]          # MGRS latitude band
    m["utm_zone"] = m["hls_tile"].str[0:2].astype(int)
    return m


# --------------------------------------------------------------------------- #
# 1. Error overlap with RF
# --------------------------------------------------------------------------- #
def error_overlap(m):
    strong, fail = 0.5, 0.1
    def cat(r):
        rg, pg = r["iou_rf"], r["iou_pr"]
        if rg >= strong and pg >= strong:
            return "both_good"
        if rg >= strong and pg < strong:
            return "RF_wins"
        if pg >= strong and rg < strong:
            return "Prithvi_wins"
        if rg < fail and pg < fail:
            return "both_fail"
        return "RF_better" if rg >= pg else "Prithvi_better"
    m["category"] = m.apply(cat, axis=1)
    tbl = m.groupby("category").agg(
        n=("event_id", "size"),
        rf_iou=("iou_rf", "mean"), pr_iou=("iou_pr", "mean"),
        rf_dice=("dice_rf", "mean"), pr_dice=("dice_pr", "mean")).reset_index()
    tbl = tbl.sort_values("n", ascending=False)
    tbl.to_csv(os.path.join(OUT, "m4_5_error_overlap.csv"), index=False)
    # also a median-split 2x2
    mr, mp = m["iou_rf"].median(), m["iou_pr"].median()
    m["rf_hi"] = m["iou_rf"] >= mr
    m["pr_hi"] = m["iou_pr"] >= mp
    crosstab = pd.crosstab(m["rf_hi"], m["pr_hi"])
    crosstab.to_csv(os.path.join(OUT, "m4_5_error_overlap_median2x2.csv"))
    print("=== 1. error overlap (IoU strong>=0.5, fail<0.1) ===")
    print(tbl.to_string(index=False))
    print("\nmedian-split 2x2 (RF strong x Prithvi strong):")
    print(crosstab.to_string())
    print(f"RF median IoU={mr:.3f}  Prithvi median IoU={mp:.3f}\n")


# --------------------------------------------------------------------------- #
# 2/3/4. Stratifications (burn-fraction, fire-year, geographic)
# --------------------------------------------------------------------------- #
def _strat(m, key, title):
    rows = []
    for k, g in m.groupby(key):
        rows.append({"group": k, "n": len(g),
                     "rf_iou": g["iou_rf"].mean(), "pr_iou": g["iou_pr"].mean(),
                     "rf_dice": g["dice_rf"].mean(), "pr_dice": g["dice_pr"].mean(),
                     "rf_auroc": g["auroc_rf"].mean(),
                     "pr_auroc": g["auroc_pr"].mean()})
    out = pd.DataFrame(rows)
    print(f"=== {title} ===")
    print(out.to_string(index=False))
    print()
    return out


def stratifications(m):
    # burn-fraction bins
    def bin_(x):
        for lo, hi, lab in BURN_BINS:
            if lo <= x < hi:
                return lab
        return "?"
    m["burn_bin"] = m["true_burn_fraction_pr"].map(bin_)
    _strat(m.sort_values("burn_bin"), "burn_bin",
           "2. burn-fraction stratification (RF vs Prithvi)").to_csv(
        os.path.join(OUT, "m4_5_burnfraction.csv"), index=False)

    _strat(m.sort_values("fire_year"), "fire_year",
           "3. fire-year stratification (non-causal)").to_csv(
        os.path.join(OUT, "m4_5_fireyear.csv"), index=False)

    _strat(m.sort_values("latband"), "latband",
           "4. geographic (MGRS latitude band) stratification").to_csv(
        os.path.join(OUT, "m4_5_geographic.csv"), index=False)


# --------------------------------------------------------------------------- #
# 5. Calibration / threshold diagnostics
# --------------------------------------------------------------------------- #
def calibration(m):
    thr = pd.read_csv(os.path.join(RESULTS_ROOT, "m4_linear_probe",
                                   "m4_thresholds.csv"))
    print("=== 5. calibration / threshold diagnostics ===")
    print("per-fold thresholds:", thr["threshold"].round(3).tolist())
    print(f"threshold spread: {thr['threshold'].min():.3f}..{thr['threshold'].max():.3f}")
    r = np.corrcoef(m["true_burn_fraction_pr"], m["pred_burn_fraction_pr"])[0, 1]
    print(f"mean true burned fraction = {m['true_burn_fraction_pr'].mean():.3f}")
    print(f"mean pred burned fraction = {m['pred_burn_fraction_pr'].mean():.3f}")
    print(f"r(true, pred burned fraction) = {r:.3f}")
    print(f"recall mean = {m['recall_pr'].mean():.3f}, precision mean = "
          f"{m['precision_pr'].mean():.3f}  (=> near-constant 'burned' collapse)")
    pd.DataFrame({
        "thresholds": thr["threshold"].tolist(),
        "inner_val_dice": thr["inner_val_dice_at_threshold"].tolist(),
    }).to_csv(os.path.join(OUT, "m4_5_thresholds.csv"), index=False)
    m[["event_id", "true_burn_fraction_pr", "pred_burn_fraction_pr",
       "precision_pr", "recall_pr"]].to_csv(
        os.path.join(OUT, "m4_5_calibration.csv"), index=False)
    print()


# --------------------------------------------------------------------------- #
# 6. Boundary-resolution diagnostic
# --------------------------------------------------------------------------- #
def _mask_path(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                        image_id + "_merged.tif").replace("_merged.tif", ".mask.tif")


def boundary_error_profile(m, img_split, n_events=576, seed=42):
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
    import rasterio
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    acc_rf = np.zeros((len(edges) - 1, 3))   # [count, sum_err_rf, sum_err_pr]
    rng = np.random.default_rng(seed)
    ids = list(m["event_id"])
    if n_events < len(ids):
        ids = rng.choice(ids, n_events, replace=False)

    for event_id in ids:
        row = m[m["event_id"] == event_id].iloc[0]
        image_id = row["image_id_rf"]
        split_dir = img_split[image_id]
        with rasterio.open(_mask_path(image_id, split_dir)) as ds:
            mask = ds.read(1).astype(np.int8)
        valid = mask >= 0
        pred_rf = np.full(mask.shape, -1, np.int8)
        pred_rf[valid] = np.load(os.path.join(RF_PRED, f"{event_id}.npy"))
        pred_pr = np.full(mask.shape, -1, np.int8)
        pred_pr[valid] = np.load(os.path.join(PR_PRED, f"{event_id}.npy"))

        burned = mask == 1
        inner = binary_erosion(burned)
        outer = binary_dilation(burned)
        ring = outer & ~inner                     # boundary ring
        d = distance_transform_edt(~ring)         # 0 at ring, grows away
        d[~valid] = -1
        err_rf = (pred_rf != mask) & valid
        err_pr = (pred_pr != mask) & valid
        for k in range(len(edges) - 1):
            sel = (d >= edges[k]) & (d < edges[k + 1])
            acc_rf[k, 0] += sel.sum()
            acc_rf[k, 1] += (err_rf & sel).sum()
            acc_rf[k, 2] += (err_pr & sel).sum()

    rows = []
    for k in range(len(edges) - 1):
        n = acc_rf[k, 0]
        lab = f"{edges[k]}-{edges[k + 1]}"
        rows.append({"dist_bin": lab, "n_pixels": int(n),
                     "rf_err_rate": acc_rf[k, 1] / n if n else np.nan,
                     "pr_err_rate": acc_rf[k, 2] / n if n else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m4_5_boundary_error.csv"), index=False)
    print("=== 6. boundary-resolution: error rate vs distance from GT boundary ===")
    print(out.to_string(index=False))
    print()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = merged()
    error_overlap(m)
    stratifications(m)
    calibration(m)
    a = _attrs()
    img_split = dict(zip(a["image_id"], a["split"]))
    boundary_error_profile(m, img_split)
    print("outputs ->", OUT)


if __name__ == "__main__":
    main()

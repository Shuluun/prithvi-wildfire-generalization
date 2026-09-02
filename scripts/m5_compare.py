"""M5 — five-model comparison and paired bootstrap inference.

Consumes the per-event OOF prediction CSVs (already computed by M2, M4, and the
M5 scripts) plus the per-event binary prediction maps, and produces:
  * m5_model_matrix.csv        — one row/event, all 5 models x {iou,dice,auprc,auroc}
  * m5_ranking.csv             — per-model mean/median/bootstrap CI
  * m5_paired_deltas.csv       — paired delta bootstrap CIs for the key pairs
  * m5_burnfraction.csv        — burned-fraction stratification per model
  * m5_calibration.csv         — threshold spread + true/pred burn-fraction r
  * m5_hardest_easiest.csv     — hardest/easiest events
  * m5_boundary_error.csv      — error rate vs distance from GT boundary

No training. Usage (from repo root):
  python scripts/m5_compare.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.evaluation.bootstrap import bootstrap_ci, summarize_events  # noqa: E402

OUT = os.path.join(RESULTS_ROOT, "m5_compare")
METRIC_COLS = ["iou", "dice", "auprc", "auroc"]

# model registry: key -> (oof csv, preds dir, display name)
MODELS = {
    "rf":        (os.path.join(RESULTS_ROOT, "m2", "event", "m2_event_oof_predictions.csv"),
                  os.path.join(RESULTS_ROOT, "m2", "event", "preds"),
                  "spectral RF"),
    "linear":    (os.path.join(RESULTS_ROOT, "m4_linear_probe", "m4_event_oof_predictions.csv"),
                  os.path.join(RESULTS_ROOT, "m4_linear_probe", "preds"),
                  "Prithvi linear"),
    "mlp":       (os.path.join(RESULTS_ROOT, "m5a_pointwise_mlp", "m5a_event_oof_predictions.csv"),
                  os.path.join(RESULTS_ROOT, "m5a_pointwise_mlp", "preds"),
                  "Prithvi MLP"),
    "decoder":   (os.path.join(RESULTS_ROOT, "m5b_spatial_decoder", "m5b_event_oof_predictions.csv"),
                  os.path.join(RESULTS_ROOT, "m5b_spatial_decoder", "preds"),
                  "Prithvi decoder"),
    "spectral":  (os.path.join(RESULTS_ROOT, "m5_spectral_cnn", "m5_spectral_event_oof_predictions.csv"),
                  os.path.join(RESULTS_ROOT, "m5_spectral_cnn", "preds"),
                  "spectral CNN"),
}
BURN_BINS = [(-1, 0.01, "<0.01"), (0.01, 0.05, "0.01-0.05"),
             (0.05, 0.20, "0.05-0.20"), (0.20, 1.01, ">=0.20")]


def load_attrs():
    a = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    return a[a["incid_type"] == "Wildfire"].copy()


def load_model_matrix():
    """Merge all 5 OOF CSVs on event_id, suffix = model key."""
    oofs = {}
    for key, (path, _pred, _name) in MODELS.items():
        df = pd.read_csv(path)
        oofs[key] = df[["event_id", "image_id"] + METRIC_COLS +
                       ["true_burn_fraction", "pred_burn_fraction"]]
    m = oofs["rf"]
    for key in ["linear", "mlp", "decoder", "spectral"]:
        m = m.merge(oofs[key], on="event_id", suffixes=("", f"_{key}"))
    # rf columns keep their base names (iou, dice, ...); non-rf get _{key} suffix
    return m


def ranking(matrix):
    rows = []
    for key, (_p, _d, name) in MODELS.items():
        for mc in METRIC_COLS:
            col = mc if key == "rf" else f"{mc}_{key}"
            vals = matrix[col].astype(float)
            lo, hi = bootstrap_ci(vals, seed=42)
            finite = vals[np.isfinite(vals)]
            rows.append({"model": key, "model_name": name, "metric": mc,
                         "n_events": int(len(finite)),
                         "mean": float(finite.mean()),
                         "median": float(finite.median()),
                         "bootstrap_ci_lo": lo, "bootstrap_ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_ranking.csv"), index=False)
    print("=== ranking (mean / median / bootstrap 95% CI) ===")
    for mc in METRIC_COLS:
        sub = out[out["metric"] == mc].sort_values("mean", ascending=False)
        print(f"\n  {mc}:")
        for _, r in sub.iterrows():
            print(f"    {r['model_name']:16s} mean={r['mean']:.4f} "
                  f"median={r['median']:.4f} CI=[{r['bootstrap_ci_lo']:.4f}, "
                  f"{r['bootstrap_ci_hi']:.4f}]")
    return out


def paired_deltas(matrix):
    pairs = [("mlp", "linear"), ("mlp", "rf"), ("decoder", "rf"),
             ("decoder", "linear"), ("decoder", "spectral"),
             ("mlp", "spectral")]
    rows = []
    for a, b in pairs:
        for mc in METRIC_COLS:
            ca = mc if a == "rf" else f"{mc}_{a}"
            cb = mc if b == "rf" else f"{mc}_{b}"
            d = (matrix[ca] - matrix[cb]).to_numpy(float)
            lo, hi = bootstrap_ci(d, seed=42)
            rows.append({"model_a": a, "model_b": b, "metric": mc,
                         "a_mean": float(matrix[ca].mean()),
                         "b_mean": float(matrix[cb].mean()),
                         "delta_mean": float(np.nanmean(d)),
                         "delta_median": float(np.nanmedian(d)),
                         "delta_ci_lo": lo, "delta_ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_paired_deltas.csv"), index=False)
    print("\n=== paired deltas (model_a - model_b) ===")
    for _, r in out.iterrows():
        print(f"  {r['model_a']:8s}-{r['model_b']:8s} {r['metric']:6s} "
              f"{r['a_mean']:.4f} - {r['b_mean']:.4f} = "
              f"{r['delta_mean']:+.4f} CI=[{r['delta_ci_lo']:+.4f}, "
              f"{r['delta_ci_hi']:+.4f}]")
    return out


def burnfraction(matrix):
    m = matrix.merge(load_attrs()[["event_id", "burned_fraction"]],
                     on="event_id", how="left")

    def bin_(x):
        for lo, hi, lab in BURN_BINS:
            if lo <= x < hi:
                return lab
        return "?"

    m["burn_bin"] = m["burned_fraction"].map(bin_)
    rows = []
    for label, g in m.groupby("burn_bin", sort=False):
        r = {"burn_bin": label, "n": len(g)}
        for key in MODELS:
            for mc in ["iou", "auroc"]:
                col = mc if key == "rf" else f"{mc}_{key}"
                r[f"{key}_{mc}"] = float(g[col].mean())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_burnfraction.csv"), index=False)
    print("\n=== burned-fraction stratification (mean IoU / AUROC) ===")
    print(out.to_string(index=False))
    return out


def calibration(matrix):
    rows = []
    for key, (path, _d, name) in MODELS.items():
        oof = pd.read_csv(path)
        thr_path = os.path.join(os.path.dirname(path),
                                f"{_tag_of(path)}_thresholds.csv")
        thr = pd.read_csv(thr_path)
        r = np.corrcoef(oof["true_burn_fraction"], oof["pred_burn_fraction"])[0, 1]
        rows.append({"model": key, "model_name": name,
                     "n_thresholds": len(thr),
                     "threshold_min": float(thr["threshold"].min()),
                     "threshold_max": float(thr["threshold"].max()),
                     "threshold_spread": float(thr["threshold"].max()
                                               - thr["threshold"].min()),
                     "mean_true_bf": float(oof["true_burn_fraction"].mean()),
                     "mean_pred_bf": float(oof["pred_burn_fraction"].mean()),
                     "r_true_pred_bf": float(r)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_calibration.csv"), index=False)
    print("\n=== calibration / threshold diagnostics ===")
    print(out.to_string(index=False))
    return out


def _tag_of(oof_path):
    base = os.path.basename(oof_path)
    return base.split("_event_oof_predictions")[0]


def hardest_easiest(matrix):
    m = matrix.copy()
    # hardest/easiest by each model's own IoU, plus the RF IoU for reference
    rows = []
    for key, (_p, _d, name) in MODELS.items():
        col = "iou" if key == "rf" else f"iou_{key}"
        hard = m.loc[m[col].idxmin()]
        easy = m.loc[m[col].idxmax()]
        rows.append({"model": name, "hardest_event": hard["event_id"],
                     "hardest_iou": float(hard[col]),
                     "easiest_event": easy["event_id"],
                     "easiest_iou": float(easy[col]),
                     "rf_iou_hardest": float(hard["iou"]),
                     "rf_iou_easiest": float(easy["iou"])})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_hardest_easiest.csv"), index=False)
    print("\n=== hardest / easiest events (by each model's IoU) ===")
    print(out.to_string(index=False))
    return out


def boundary_error(matrix, n_events=576, seed=42):
    """Error rate vs distance from the GT burn boundary, for every model."""
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
    import rasterio
    edges = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    keys = list(MODELS)
    n_models = len(keys)
    acc = np.zeros((len(edges) - 1, 1 + n_models))  # [count, err_rf, ...]
    attrs = load_attrs()
    img_split = dict(zip(attrs["image_id"], attrs["split"]))
    rng = np.random.default_rng(seed)
    ids = list(matrix["event_id"])
    if n_events < len(ids):
        ids = rng.choice(ids, n_events, replace=False)

    def mask_path(image_id, split_dir):
        return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                            image_id + "_merged.tif").replace("_merged.tif",
                                                              ".mask.tif")

    for event_id in ids:
        row = matrix[matrix["event_id"] == event_id].iloc[0]
        image_id = row["image_id"]
        with rasterio.open(mask_path(image_id, img_split[image_id])) as ds:
            mask = ds.read(1).astype(np.int8)
        valid = mask >= 0
        burned = mask == 1
        inner = binary_erosion(burned)
        outer = binary_dilation(burned)
        ring = outer & ~inner
        d = distance_transform_edt(~ring)
        d[~valid] = -1
        for k, key in enumerate(keys):
            pred_dir = MODELS[key][1]
            pred = np.full(mask.shape, -1, np.int8)
            pred[valid] = np.load(os.path.join(pred_dir, f"{event_id}.npy"))
            err = (pred != mask) & valid
            for b in range(len(edges) - 1):
                sel = (d >= edges[b]) & (d < edges[b + 1])
                if k == 0:
                    acc[b, 0] += sel.sum()
                acc[b, 1 + k] += (err & sel).sum()

    rows = []
    for b in range(len(edges) - 1):
        n = acc[b, 0]
        r = {"dist_bin": f"{edges[b]}-{edges[b+1]}", "n_pixels": int(n)}
        for k, key in enumerate(keys):
            r[f"{key}_err_rate"] = acc[b, 1 + k] / n if n else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "m5_boundary_error.csv"), index=False)
    print("\n=== boundary/interior error rate (boundary -> interior) ===")
    cols = ["dist_bin"] + [f"{k}_err_rate" for k in keys]
    print(out[cols].to_string(index=False))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    matrix = load_model_matrix()
    # ensure all 5 models cover the same 576 events
    for key in MODELS:
        col = "iou" if key == "rf" else f"iou_{key}"
        assert matrix[col].notna().sum() == 576, (key, matrix[col].notna().sum())
    matrix.to_csv(os.path.join(OUT, "m5_model_matrix.csv"), index=False)
    print(f"model matrix: {len(matrix)} events x "
          f"{1 + len(MODELS) * len(METRIC_COLS)} metric cols")
    ranking(matrix)
    paired_deltas(matrix)
    burnfraction(matrix)
    calibration(matrix)
    hardest_easiest(matrix)
    boundary_error(matrix)
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()

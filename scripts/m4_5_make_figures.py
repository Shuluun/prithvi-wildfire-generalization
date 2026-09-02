"""M4.5 figures — boundary-resolution error curve + visual failure-case panels.

Consumes the M4.5 CSVs and the per-event binary maps (RF + Prithvi-linear). No
training. Outputs to results/m4_5/.

Usage (from repo root):
  python scripts/m4_5_make_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402

OUT = os.path.join(RESULTS_ROOT, "m4_5")
RF_OOF = os.path.join(RESULTS_ROOT, "m2", "event", "m2_event_oof_predictions.csv")
PR_OOF = os.path.join(RESULTS_ROOT, "m4_linear_probe", "m4_event_oof_predictions.csv")
RF_PRED = os.path.join(RESULTS_ROOT, "m2", "event", "preds")
PR_PRED = os.path.join(RESULTS_ROOT, "m4_linear_probe", "preds")

RF_COL = "#2c7fb8"
PR_COL = "#d7191c"
GT_COLORS = {1: "#d7191c", 0: "#bdbdbd", -1: "#000000"}


def _mask_path(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                        image_id + "_merged.tif").replace("_merged.tif", ".mask.tif")


def _load_rgb_mask(image_id, split_dir):
    mp = os.path.join(HLS_BURN_SCARS_DIR, split_dir, image_id + "_merged.tif")
    with rasterio.open(mp) as ds:
        img = ds.read().astype(np.float32)
    with rasterio.open(mp.replace("_merged.tif", ".mask.tif")) as ds:
        mask = ds.read(1)
    rgb = np.stack([img[2], img[1], img[0]], axis=-1)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    disp = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return disp, mask


def _mask_rgb(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=float)
    for val, col in GT_COLORS.items():
        c = np.array([int(col[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        out[mask == val] = c
    return out


def _pred_map(event_id, pred_dir, mask):
    p = np.full(mask.shape, -1, np.int8)
    p[mask >= 0] = np.load(os.path.join(pred_dir, f"{event_id}.npy"))
    return p


def fig_boundary(out):
    b = pd.read_csv(os.path.join(OUT, "m4_5_boundary_error.csv"))
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, b["rf_err_rate"], "-o", color=RF_COL, lw=1.8, ms=5,
            label="spectral RF")
    ax.plot(x, b["pr_err_rate"], "-o", color=PR_COL, lw=1.8, ms=5,
            label="Prithvi linear")
    ax.set_xticks(x)
    ax.set_xticklabels(b["dist_bin"], rotation=45, fontsize=8)
    ax.set_xlabel("distance from ground-truth burn boundary (px)")
    ax.set_ylabel("pixel error rate")
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#636363", alpha=0.2, lw=0.5)
    fig.suptitle("Error rate vs distance from burn boundary — RF vs Prithvi linear")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_m4_5_boundary_error.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _categories(m):
    strong, fail = 0.5, 0.1
    m["cat"] = "RF_better"
    m.loc[(m.iou_rf >= strong) & (m.iou_pr < strong), "cat"] = "RF_wins"
    m.loc[(m.iou_pr >= strong) & (m.iou_rf < strong), "cat"] = "Prithvi_wins"
    m.loc[(m.iou_rf >= strong) & (m.iou_pr >= strong), "cat"] = "both_good"
    m.loc[(m.iou_rf < fail) & (m.iou_pr < fail), "cat"] = "both_fail"
    m.loc[(m.iou_pr >= m.iou_rf) & (m.cat == "RF_better"), "cat"] = "Prithvi_better"
    return m


def fig_panels(m, img_split, out, k=2):
    rf = pd.read_csv(RF_OOF); pr = pd.read_csv(PR_OOF)
    mm = rf.merge(pr, on="event_id", suffixes=("_rf", "_pr"))
    mm = _categories(mm)

    picks = {
        "RF >> Prithvi": mm[mm.cat == "RF_wins"].sort_values(
            "iou_pr", ascending=True).head(k),
        "Prithvi >> RF": mm[mm.cat.isin(["Prithvi_wins", "Prithvi_better"])]
            .sort_values("iou_rf", ascending=True).head(k),
        "both good": mm[mm.cat == "both_good"].sort_values(
            "iou_pr", ascending=False).head(k),
        "both poor": mm[mm.cat == "both_fail"].sort_values(
            "iou_rf", ascending=True).head(k),
        "low burn": mm.sort_values("true_burn_fraction_rf", ascending=True).head(k),
    }

    n_rows = len(picks)
    fig, axes = plt.subplots(n_rows, 4 * k, figsize=(4 * k * 1.6, n_rows * 1.9))
    for ri, (label, sub) in enumerate(picks.items()):
        for j, (_, row) in enumerate(sub.iterrows()):
            image_id = row["image_id_rf"]
            disp, mask = _load_rgb_mask(image_id, img_split[image_id])
            p_rf = _pred_map(row["event_id"], RF_PRED, mask)
            p_pr = _pred_map(row["event_id"], PR_PRED, mask)
            axs = [axes[ri, 4 * j], axes[ri, 4 * j + 1],
                   axes[ri, 4 * j + 2], axes[ri, 4 * j + 3]]
            axs[0].imshow(disp); axs[0].set_axis_off()
            axs[1].imshow(_mask_rgb(mask)); axs[1].set_axis_off()
            axs[2].imshow(_mask_rgb(p_rf)); axs[2].set_axis_off()
            axs[3].imshow(_mask_rgb(p_pr)); axs[3].set_axis_off()
            if j == 0:
                axs[0].set_title(
                    f"{label}\nRF IoU={row['iou_rf']:.2f}  P IoU={row['iou_pr']:.2f}",
                    fontsize=8, loc="left")
            if ri == 0:
                for ax, t in zip(axs, ["RGB", "GT", "RF", "Prithvi"]):
                    ax.set_title(t, fontsize=8)
    fig.suptitle("Frozen Prithvi linear probe vs spectral RF — representative panels")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_m4_5_failure_panels.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(RF_OOF).merge(pd.read_csv(PR_OOF), on="event_id",
                                  suffixes=("_rf", "_pr"))
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    img_split = dict(zip(attrs["image_id"], attrs["split"]))
    figs = [fig_boundary(OUT), fig_panels(m, img_split, OUT)]
    print("figures written:")
    for f in figs:
        print(" ", f)


if __name__ == "__main__":
    main()

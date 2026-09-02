"""M2 figures for the spectral baseline (PRIMARY protocol).

Reads results/m2/event/m2_event_oof_predictions.csv + the saved per-event
binary prediction maps and produces:
  1. per-event burned-IoU distribution
  2. per-event burned-Dice distribution
  3. IoU vs true burned fraction
  4. representative predictions (RGB / ground truth / RF binary) for
     low / median / high IoU events (not hand-picked good examples)

Usage (from repo root):  python scripts/m2_make_figures.py
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

HUE = "#2c7fb8"      # single sequential hue for magnitude
MUTED = "#636363"    # recessive grid / axes
GT_COLORS = {1: "#d7191c", 0: "#bdbdbd", -1: "#000000"}  # burned/unburned/missing


def load_rgb_mask(image_id, split_dir):
    mp = os.path.join(HLS_BURN_SCARS_DIR, split_dir, image_id + "_merged.tif")
    with rasterio.open(mp) as ds:
        img = ds.read().astype(np.float32)  # (6, H, W)
    with rasterio.open(mp.replace("_merged.tif", ".mask.tif")) as mds:
        mask = mds.read(1)
    rgb = np.stack([img[2], img[1], img[0]], axis=-1)  # B04,B03,B02 -> R,G,B
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    disp = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return disp, mask


def mask_rgb(mask):
    """Map label mask {-1,0,1} to an RGB display image."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=float)
    for val, col in GT_COLORS.items():
        c = np.array([int(col[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        out[mask == val] = c
    return out


def fig_distributions(oof, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, name in [(axes[0], "iou", "burned IoU"),
                          (axes[1], "dice", "burned Dice/F1")]:
        vals = oof[col].dropna()
        ax.hist(vals, bins=30, color=HUE, edgecolor="white", linewidth=0.5)
        ax.axvline(vals.median(), color="#000000", lw=1.2, ls="--")
        ax.text(0.97, 0.95, f"median={vals.median():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_xlabel(name)
        ax.set_ylabel("events")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(color=MUTED)
    fig.suptitle(f"Spectral RF — per-event distribution (N={len(oof)} events)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_m2_distributions.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_iou_vs_burnfraction(oof, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(oof["true_burn_fraction"], oof["iou"], s=18, color=HUE,
               alpha=0.55, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("true burned fraction (log)")
    ax.set_ylabel("burned IoU")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(color=MUTED)
    ax.grid(color=MUTED, alpha=0.2, lw=0.5)
    fig.suptitle("Spectral RF — IoU vs burned fraction")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_m2_iou_vs_burnfraction.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _representative_events(oof, k=3):
    s = oof.sort_values("iou").reset_index(drop=True)
    n = len(s)
    low = s.iloc[:k]
    high = s.iloc[-k:]
    mid_start = max(0, (n - k) // 2)
    med = s.iloc[mid_start:mid_start + k]
    return low, med, high


def fig_representative(oof, img_split, out_dir, k=3):
    low, med, high = _representative_events(oof, k)
    groups = [("low IoU", low), ("median IoU", med), ("high IoU", high)]
    fig, axes = plt.subplots(len(groups), 3 * k, figsize=(3 * k * 1.6,
                                                         len(groups) * 1.9))
    for gi, (label, sub) in enumerate(groups):
        for j, (_, row) in enumerate(sub.iterrows()):
            disp, mask = load_rgb_mask(row["image_id"],
                                       img_split[row["image_id"]])
            pred_flat = np.load(os.path.join(
                out_dir, "preds", f"{row['event_id']}.npy"))
            # pred .npy holds valid-pixel predictions only; rebuild full map
            pred = np.full(mask.shape, -1, dtype=np.int8)
            pred[mask >= 0] = pred_flat
            ax_rgb, ax_gt, ax_pd = (axes[gi, 3 * j], axes[gi, 3 * j + 1],
                                    axes[gi, 3 * j + 2])
            ax_rgb.imshow(disp); ax_rgb.set_axis_off()
            ax_gt.imshow(mask_rgb(mask)); ax_gt.set_axis_off()
            ax_pd.imshow(mask_rgb(pred)); ax_pd.set_axis_off()
            if j == 0:
                ax_rgb.set_title(f"{label}\nIoU={row['iou']:.3f}",
                                 fontsize=8, loc="left")
            if gi == 0:
                for ax, t in [(ax_rgb, "RGB"), (ax_gt, "ground truth"),
                              (ax_pd, "RF pred")]:
                    ax.set_title(t, fontsize=8)
    fig.suptitle("Spectral RF — representative predictions (low / median / high IoU)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_m2_representative_predictions.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_paired_delta(m2_dir, out_dir):
    """Per-event PRIMARY-vs-SPATIAL delta distributions (4 metrics)."""
    d = pd.read_csv(os.path.join(m2_dir, "m2_paired_deltas_per_event.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, m in zip(axes.ravel(), ["iou", "dice", "auprc", "auroc"]):
        col = f"delta_{m}"
        vals = d[col].dropna()
        ax.hist(vals, bins=40, color=HUE, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="#000000", lw=1.0)
        ax.axvline(vals.mean(), color="#d7191c", lw=1.2, ls="--")
        ax.text(0.97, 0.95, f"mean={vals.mean():+.4f}\nmedian={vals.median():+.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_xlabel(f"$\\Delta$ {m} (spatial − primary)")
        ax.set_ylabel("events")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(color=MUTED)
    fig.suptitle("Spectral RF — paired spatial-clean effect (per-event delta)")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_m2_paired_delta.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    out_dir = os.path.join(RESULTS_ROOT, "m2", "event")
    m2_dir = os.path.join(RESULTS_ROOT, "m2")
    oof = pd.read_csv(os.path.join(out_dir, "m2_event_oof_predictions.csv"))
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    img_split = dict(zip(attrs["image_id"], attrs["split"]))
    figs = [
        fig_distributions(oof, out_dir),
        fig_iou_vs_burnfraction(oof, out_dir),
        fig_representative(oof, img_split, out_dir),
        fig_paired_delta(m2_dir, m2_dir),
    ]
    print("figures written:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

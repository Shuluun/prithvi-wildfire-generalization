"""M6 final figure set for the wildfire cross-event representation study.

Reads only COMMITTED results (results/m5_compare/*.csv, the per-event OOF CSVs,
and per-event prediction maps) and produces the six figures requested for the
README / research note. No training, no new model results.

  fig1  study design / evaluation schematic
  fig2  five-model primary performance (event-level distributions + mean/CI)
  fig3  paired decoder-vs-spectral (scatter + per-event delta)
  fig4  performance vs burned fraction
  fig5  representative qualitative cases (not cherry-picked)
  fig6  diagnostic summary: linear -> MLP -> spatial decoder

Usage (from repo root):  python scripts/m6_make_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.evaluation.bootstrap import bootstrap_ci  # noqa: E402

OUT_DIR = os.path.join(RESULTS_ROOT, "figures", "final")
CMP = os.path.join(RESULTS_ROOT, "m5_compare")

# model registry: key -> (display name, color). Spectral family = cool hues,
# Prithvi readouts = warm hues. Colors chosen for pairwise distinctness + direct
# labels (color is never the sole encoding).
MODELS = {
    "rf":        ("spectral RF",      "#2c7fb8"),
    "spectral":  ("spectral CNN",     "#2ca25f"),
    "linear":    ("Prithvi linear",   "#e6ab02"),
    "mlp":       ("Prithvi MLP",      "#d95f02"),
    "decoder":   ("Prithvi decoder",  "#e7298a"),
}
ORDER = ["rf", "spectral", "linear", "mlp", "decoder"]
MUTED = "#636363"
GT_COLORS = {1: "#d7191c", 0: "#bdbdbd", -1: "#000000"}


def _col(key, metric):
    return metric if key == "rf" else f"{metric}_{key}"


def load_matrix():
    return pd.read_csv(os.path.join(CMP, "m5_model_matrix.csv"))


def load_attrs():
    a = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    return a[a["incid_type"] == "Wildfire"].copy()


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(color=MUTED)
    ax.grid(color=MUTED, alpha=0.18, lw=0.5, axis="y")


def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Figure 1 — study design schematic
# ----------------------------------------------------------------------------
def fig1_study_design():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.35, 4.4, 2.4, 1.0, "HLS BurnScars\n512×512 chips", "#e3e3e3"),
        (0.35, 2.6, 2.4, 1.0, "MTBS event\nreconstruction", "#e3e3e3"),
        (3.15, 3.5, 2.4, 1.0, "576 matched\nWildfire events", "#deebf7"),
        (6.05, 3.5, 2.9, 1.0, "shared event-disjoint\n5-fold CV (seed 42)", "#deebf7"),
        (9.15, 3.5, 0.001, 0.001, "", "#ffffff"),  # placeholder (unused)
    ]
    # two downstream branches
    branch_spectral = (6.05, 4.6, 3.6, 1.0, "spectral models\nRF  ·  CNN", "#ccece6")
    branch_prithvi = (6.05, 2.4, 3.6, 1.0, "frozen-Prithvi models\nlinear · MLP · decoder", "#fdd0a2")
    boxes = boxes[:4] + [branch_spectral, branch_prithvi]

    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    linewidth=1.1, edgecolor="#4d4d4d", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#4d4d4d"))

    arrow(1.55, 4.4, 1.55, 3.6)                 # chips -> reconstruction
    arrow(2.75, 3.1, 3.15, 3.85)                # reconstruction -> events
    arrow(4.35, 4.0, 6.05, 4.0)                 # events -> shared folds
    arrow(7.5, 4.5, 7.5, 4.0)                   # folds -> spectral branch
    arrow(7.5, 3.5, 7.5, 3.4)                   # folds -> prithvi branch
    ax.text(1.55, 4.62, "geometry + time\n+ MTBS overlap", ha="center", fontsize=7.3,
            color=MUTED)
    ax.text(5.4, 2.0, "1 OOF prediction per event per model → paired event-level comparison",
            ha="center", fontsize=8.6, style="italic")
    return _save(fig, "fig1_study_design.png")


# ----------------------------------------------------------------------------
# Figure 2 — five-model primary performance (distributions + mean/CI)
# ----------------------------------------------------------------------------
def fig2_performance(matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in [(axes[0], "iou"), (axes[1], "auroc")]:
        data, colors, means, los, his = [], [], [], [], []
        for key in ORDER:
            vals = matrix[_col(key, metric)].astype(float)
            finite = vals[np.isfinite(vals)]
            data.append(finite.to_numpy())
            colors.append(MODELS[key][1])
            lo, hi = bootstrap_ci(finite, seed=42)
            means.append(finite.mean())
            los.append(lo)
            his.append(hi)
        bp = ax.boxplot(data, positions=range(len(ORDER)), widths=0.55,
                        showfliers=False, patch_artist=True, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for i, d in enumerate(data):
            x = np.random.default_rng(i).uniform(i - 0.22, i + 0.22, len(d))
            ax.scatter(x, d, s=6, color=colors[i], alpha=0.28, linewidths=0, zorder=2)
        xs = np.arange(len(ORDER))
        ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los),
                                     np.array(his) - np.array(means)],
                    fmt="o", color="black", ms=5, lw=1.4, capsize=3, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODELS[k][0] for k in ORDER], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(f"event-level {metric.upper()}")
        ax.set_ylim(0, 1.02)
        _style_ax(ax)
    axes[0].set_title("burned IoU  (dots = events, ● = mean ± 95% bootstrap CI)")
    axes[1].set_title("AUROC  (dots = events, ● = mean ± 95% bootstrap CI)")
    fig.suptitle("Five-model primary comparison (576 events, event-disjoint K5)")
    fig.tight_layout()
    return _save(fig, "fig2_performance.png")


# ----------------------------------------------------------------------------
# Figure 3 — paired decoder vs spectral
# ----------------------------------------------------------------------------
def fig3_paired(matrix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    s = matrix["iou_spectral"].astype(float)
    d = matrix["iou_decoder"].astype(float)
    ax = axes[0]
    ax.scatter(s, d, s=16, color=MODELS["decoder"][1], alpha=0.5, linewidths=0)
    lim = (0, 1)
    ax.plot(lim, lim, color="black", lw=1.0, ls="--")
    ax.set_xlabel("spectral CNN event IoU")
    ax.set_ylabel("Prithvi decoder event IoU")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.04, 0.95, f"{int((d > s).sum())}/{len(d)} events above diagonal",
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    _style_ax(ax)

    delta = (d - s).to_numpy()
    ax = axes[1]
    ax.hist(delta, bins=50, color=MODELS["decoder"][1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1.0)
    ax.axvline(delta.mean(), color="#d7191c", lw=1.3, ls="--")
    lo, hi = bootstrap_ci(delta, seed=42)
    ax.text(0.97, 0.95,
            f"mean = {delta.mean():+.3f}\n95% CI = [{lo:+.3f}, {hi:+.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.set_xlabel("Δ event IoU (decoder − spectral CNN)")
    ax.set_ylabel("events")
    _style_ax(ax)
    fig.suptitle("Paired per-event comparison: matched spatial decoders, input representation differs")
    fig.tight_layout()
    return _save(fig, "fig3_paired_decoder_vs_spectral.png")


# ----------------------------------------------------------------------------
# Figure 4 — performance vs burned fraction
# ----------------------------------------------------------------------------
def fig4_burnfraction(matrix):
    burn = pd.read_csv(os.path.join(CMP, "m5_burnfraction.csv"))
    bins = ["<0.01", "0.01-0.05", "0.05-0.20", ">=0.20"]
    n_by_bin = {"<0.01": 1, "0.01-0.05": 283, "0.05-0.20": 179, ">=0.20": 113}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # left: binned mean IoU (grouped bars)
    ax = axes[0]
    x = np.arange(len(bins))
    w = 0.16
    for i, key in enumerate(ORDER):
        col = "rf_iou" if key == "rf" else f"{key}_iou"
        vals = [burn.loc[burn["burn_bin"] == b, col].iloc[0] for b in bins]
        ax.bar(x + (i - 2) * w, vals, w, color=MODELS[key][1],
               alpha=0.85, label=MODELS[key][0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n_by_bin[b]})" for b in bins], fontsize=8)
    ax.set_ylabel("mean event IoU")
    ax.set_title("mean IoU by burned-fraction bin")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style_ax(ax)

    # right: per-event scatter, spectral CNN vs decoder, log-x
    ax = axes[1]
    bf = matrix["true_burn_fraction"].astype(float)
    ax.scatter(bf, matrix["iou_spectral"], s=12, color=MODELS["spectral"][1],
               alpha=0.45, linewidths=0, label="spectral CNN")
    ax.scatter(bf, matrix["iou_decoder"], s=12, color=MODELS["decoder"][1],
               alpha=0.45, linewidths=0, label="Prithvi decoder")
    ax.set_xscale("log")
    ax.set_xlabel("true burned fraction (log)")
    ax.set_ylabel("event IoU")
    ax.set_title("per-event IoU vs burned fraction")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style_ax(ax)
    fig.suptitle("Low-burn events are hard, and the models differ most there")
    fig.tight_layout()
    return _save(fig, "fig4_burn_fraction.png")


# ----------------------------------------------------------------------------
# Figure 5 — representative qualitative cases
# ----------------------------------------------------------------------------
def _load_rgb_mask(image_id, split_dir):
    mp = os.path.join(HLS_BURN_SCARS_DIR, split_dir, image_id + "_merged.tif")
    with rasterio.open(mp) as ds:
        img = ds.read().astype(np.float32)
    with rasterio.open(mp.replace("_merged.tif", ".mask.tif")) as mds:
        mask = mds.read(1)
    rgb = np.stack([img[2], img[1], img[0]], axis=-1)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    disp = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return disp, mask


def _mask_rgb(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=float)
    for val, col in GT_COLORS.items():
        c = np.array([int(col[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
        out[mask == val] = c
    return out


def _load_pred(pred_dir, event_id, mask):
    pred = np.full(mask.shape, -1, dtype=np.int8)
    pred[mask >= 0] = np.load(os.path.join(pred_dir, f"{event_id}.npy"))
    return pred


def _pick_case(matrix, mode):
    s = matrix["iou_spectral"].astype(float)
    d = matrix["iou_decoder"].astype(float)
    bf = matrix["true_burn_fraction"].astype(float)
    if mode == "spectral_wins":
        m = matrix[(s > 0.5) & (d < 0.15)].copy()
        m = m.sort_values("iou_spectral", ascending=False)
    elif mode == "both_hard":
        m = matrix[(bf > 0) & (s < 0.2) & (d < 0.2)].copy()
        m = m.sort_values("true_burn_fraction")
    elif mode == "prithvi_wins":
        m = matrix[(d - s > 0.1) & (d > 0.3)].copy()
        m = m.sort_values("iou_decoder", ascending=False)
    elif mode == "small_burn":
        m = matrix[(bf < 0.05)].copy()
        m = m.sort_values("iou_decoder")
    else:
        raise ValueError(mode)
    return m.iloc[0]


def fig5_qualitative(matrix, attrs):
    img_split = dict(zip(attrs["image_id"], attrs["split"]))
    pred_spectral = os.path.join(RESULTS_ROOT, "m5_spectral_cnn", "preds")
    pred_decoder = os.path.join(RESULTS_ROOT, "m5b_spatial_decoder", "preds")

    cases = [
        ("spectral ≫ Prithvi", "spectral_wins"),
        ("both difficult", "both_hard"),
        ("rare Prithvi-favorable", "prithvi_wins"),
        ("small-burn failure", "small_burn"),
    ]
    ncol = 4  # RGB, GT, spectral CNN, Prithvi decoder
    fig, axes = plt.subplots(len(cases), ncol, figsize=(ncol * 1.9, len(cases) * 1.9))
    col_titles = ["RGB", "ground truth", "spectral CNN", "Prithvi decoder"]

    for ri, (label, mode) in enumerate(cases):
        row = _pick_case(matrix, mode)
        event_id, image_id = row["event_id"], row["image_id"]
        disp, mask = _load_rgb_mask(image_id, img_split[image_id])
        p_sp = _load_pred(pred_spectral, event_id, mask)
        p_dc = _load_pred(pred_decoder, event_id, mask)
        panels = [disp, _mask_rgb(mask), _mask_rgb(p_sp), _mask_rgb(p_dc)]
        for ci, im in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(im)
            ax.set_axis_off()
        axes[ri, 0].set_title(
            f"{label}\nIoU spec={row['iou_spectral']:.2f} / dec={row['iou_decoder']:.2f}\n"
            f"burn={row['true_burn_fraction']:.3f}",
            fontsize=7.4, loc="left")
        if ri == 0:
            for ci, t in enumerate(col_titles):
                axes[ri, ci].set_title(t, fontsize=8.5)
    fig.suptitle("Representative cases (selected by criterion, not hand-picked for quality)")
    fig.tight_layout()
    return _save(fig, "fig5_qualitative_cases.png")


# ----------------------------------------------------------------------------
# Figure 6 — diagnostic: linear -> MLP -> decoder does not recover transfer
# ----------------------------------------------------------------------------
def fig6_diagnostic(matrix):
    keys = ["linear", "mlp", "decoder", "spectral"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in [(axes[0], "iou"), (axes[1], "auroc")]:
        means, los, his = [], [], []
        for key in keys:
            vals = matrix[_col(key, metric)].astype(float)
            finite = vals[np.isfinite(vals)]
            lo, hi = bootstrap_ci(finite, seed=42)
            means.append(finite.mean())
            los.append(lo)
            his.append(hi)
        xs = np.arange(len(keys))
        cols = [MODELS[k][1] for k in keys]
        ax.bar(xs, means, 0.6, color=cols, alpha=0.85)
        ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los),
                                     np.array(his) - np.array(means)],
                    fmt="none", ecolor="black", lw=1.2, capsize=4)
        for x, m in zip(xs, means):
            ax.text(x, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODELS[k][0] for k in keys], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(f"event-level {metric.upper()}")
        ax.set_ylim(0, 1.05 if metric == "auroc" else 0.62)
        _style_ax(ax)
    axes[0].set_title("IoU: readout capacity leaves it flat")
    axes[1].set_title("AUROC: MLP helps ranking, decoder does not localize")
    fig.suptitle("Additional readout capacity does not recover RF-level generalization")
    fig.tight_layout()
    return _save(fig, "fig6_diagnostic_readout.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    matrix = load_matrix()
    attrs = load_attrs()
    figs = [
        fig1_study_design(),
        fig2_performance(matrix),
        fig3_paired(matrix),
        fig4_burnfraction(matrix),
        fig5_qualitative(matrix, attrs),
        fig6_diagnostic(matrix),
    ]
    print("figures written:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()

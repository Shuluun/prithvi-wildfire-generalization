"""M4b — linear spatial probe on frozen Prithvi features (representation control).

Deliberately weak probe: frozen layers [5,11,17,23] -> concatenate channels
(4096) -> single learned 1x1 convolution to a burned logit -> bilinear upsample
to the native 512x512 grid. No spatial conv, no UNet, no morphology, no
augmentation.

Reuses the M2 frozen inner-split manifests and the M2 deterministic training-pixel
rule (max 2048 valid pixels/train chip, same global seed + image_id sampling).
The encoder is frozen and was run ONCE during M4a caching; the probe trains only
on cached features. Threshold selection matches M2 (maximize burned Dice on inner
validation over the same grid). Outer test is touched once per fold.

Usage (from repo root):
  python scripts/m4b_linear_probe.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.data.splits import SPLITS_DIR  # noqa: E402
from src.evaluation.bootstrap import bootstrap_ci, summarize_events  # noqa: E402
from src.evaluation.metrics import per_chip_metrics, select_threshold  # noqa: E402
from src.features.spectral import chip_rng, sample_valid_indices  # noqa: E402
from src.models import prithvi  # noqa: E402

SEED = 42
CAP = 2048
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FEAT = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache", "features")
OUT_ROOT = os.path.join(RESULTS_ROOT, "m4_linear_probe")

METRIC_COLS = ["iou", "dice", "precision", "recall", "auprc", "auroc"]

# fixed training config (user spec §8)
OPT = dict(lr=1e-3, weight_decay=1e-4, max_epochs=50, patience=5)


class LinearProbe(nn.Module):
    def __init__(self, in_ch=4096):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        # small init for a stable linear probe
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, feat32):  # feat32: (1, 4096, 32, 32) -> (1, 1, 512, 512) logits
        logits = self.conv(feat32)
        return F.interpolate(logits, size=(512, 512), mode="bilinear",
                             align_corners=False)

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_features(image_id):
    """Load cached frozen features -> (1, 4096, 32, 32) fp32 on device."""
    a = np.load(os.path.join(CACHE_FEAT, f"{image_id}.npy"))  # (4,1024,32,32) fp16
    a = a.astype(np.float32).reshape(1, -1, a.shape[2], a.shape[3])
    return torch.from_numpy(a).to(DEVICE)


def mask_path_of(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                        image_id + "_merged.tif").replace("_merged.tif", ".mask.tif")


def load_mask(image_id, split_dir):
    import rasterio
    with rasterio.open(mask_path_of(image_id, split_dir)) as ds:
        return ds.read(1).astype(np.int8)  # (512,512)


def sampled_pixel_logits_labels(probe, image_id, split_dir, rng, cap=CAP):
    """One chip: deterministic sampled-pixel logits + labels for training."""
    mask = load_mask(image_id, split_dir)
    idx = sample_valid_indices(mask.ravel(), cap, rng)
    rows, cols = idx // mask.shape[1], idx % mask.shape[1]
    y = mask.ravel()[idx].astype(np.int64)
    feat = load_features(image_id)
    logits = probe(feat)  # (1,1,512,512)
    logits_s = logits[0, 0, rows, cols]
    return logits_s, torch.from_numpy(y).to(DEVICE)


def eval_chip_logits(probe, image_id, split_dir):
    """All valid-pixel logits + labels for one chip (val/test)."""
    mask = load_mask(image_id, split_dir)
    valid = mask >= 0
    feat = load_features(image_id)
    logits = probe(feat)[0, 0]  # (512,512)
    yv = mask[valid].astype(np.int64)
    lv = logits[valid].detach().cpu().numpy().astype(np.float32)
    return lv, yv


def train_probe(train_rows, val_rows, img_split, pos_weight):
    probe = LinearProbe().to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=OPT["lr"],
                            weight_decay=OPT["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],
                                                           device=DEVICE))
    best = {"epoch": -1, "val_dice": -1.0, "state": None, "loss": None}
    patience_left = OPT["patience"]

    for epoch in range(1, OPT["max_epochs"] + 1):
        probe.train()
        losses = []
        for image_id in train_rows["image_id"]:
            rng = chip_rng(SEED, image_id)
            logits_s, y = sampled_pixel_logits_labels(
                probe, image_id, img_split[image_id], rng, CAP)
            if len(y) == 0:
                continue
            loss = loss_fn(logits_s, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")

        # early-stopping monitor: inner-val Dice (best threshold over grid)
        val_dice, _ = fold_val_dice(probe, val_rows, img_split)
        if val_dice > best["val_dice"]:
            best = {"epoch": epoch, "val_dice": val_dice,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in probe.state_dict().items()},
                    "loss": mean_loss}
            patience_left = OPT["patience"]
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    probe.load_state_dict(best["state"])
    return probe, best


def fold_val_dice(probe, val_rows, img_split):
    yt, yp = [], []
    for image_id in val_rows["image_id"]:
        lv, yv = eval_chip_logits(probe, image_id, img_split[image_id])
        if len(yv) == 0:
            continue
        yt.append(yv); yp.append(lv)
    if not yt:
        return 0.0, None
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    thr, dice = select_threshold(yt, _sigmoid(yp), THRESHOLD_GRID)
    return dice, thr


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def main():
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    assert set(inner["event_id"]) == set(primary["event_id"])

    os.makedirs(OUT_ROOT, exist_ok=True)
    pred_dir = os.path.join(OUT_ROOT, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    oof_all, fold_rows, thr_rows, train_rows_out = [], [], [], []
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        train_rows = grp[grp["role"] == "train"]
        val_rows = grp[grp["role"] == "val"]
        test_rows = grp[grp["role"] == "test"]

        # class weight from inner-train supervised pixels (balanced)
        n_pos = n_neg = 0
        for image_id in train_rows["image_id"]:
            m = load_mask(image_id, img_split[image_id])
            n_pos += int((m == 1).sum()); n_neg += int((m == 0).sum())
        pos_weight = n_neg / n_pos if n_pos else 1.0

        probe, best = train_probe(train_rows, val_rows, img_split, pos_weight)
        # threshold on inner-val (maximize Dice) at best epoch
        val_dice, thr = fold_val_dice(probe, val_rows, img_split)

        for image_id, event_id in zip(test_rows["image_id"],
                                      test_rows["event_id"]):
            lv, yv = eval_chip_logits(probe, image_id, img_split[image_id])
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
        train_rows_out.append({"fold": f, "n_trainable": probe.n_trainable,
                               "pos_weight": pos_weight,
                               "best_epoch": best["epoch"],
                               "train_loss": best["loss"]})
        print(f"[m4b] fold {f}: train={len(train_rows)} val={len(val_rows)} "
              f"test={len(test_rows)} thr={thr:.4f} val_dice={val_dice:.4f} "
              f"best_epoch={best['epoch']} loss={best['loss']:.4f} "
              f"trainable={probe.n_trainable}")

    oof = pd.DataFrame(oof_all)
    assert oof["event_id"].is_unique and len(oof) == len(primary)
    col_order = ["event_id", "image_id", "fold", "threshold",
                 "iou", "dice", "precision", "recall", "auprc", "auroc",
                 "true_burn_fraction", "pred_burn_fraction", "n_valid"]
    oof = oof[col_order].sort_values("event_id").reset_index(drop=True)
    oof.to_csv(os.path.join(OUT_ROOT, "m4_event_oof_predictions.csv"),
               index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_ROOT, "m4_fold_summary.csv"),
                                   index=False)
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUT_ROOT, "m4_thresholds.csv"),
                                   index=False)
    pd.DataFrame(train_rows_out).to_csv(
        os.path.join(OUT_ROOT, "m4_training_summary.csv"), index=False)
    summary = summarize_events(oof, METRIC_COLS)
    summary.to_csv(os.path.join(OUT_ROOT, "m4_metrics_summary.csv"), index=False)

    with open(os.path.join(OUT_ROOT, "m4_config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "probe": "linear 1x1 conv (4096->1) + bilinear upsample",
            "selected_layers": prithvi.SELECTED_LAYERS,
            "trainable_params": int(train_rows_out[0]["n_trainable"]),
            "seed": SEED, "cap_pixels_per_train_chip": CAP,
            "opt": OPT, "threshold_grid": THRESHOLD_GRID.tolist(),
            "n_primary_events": int(len(primary)),
            "preprocessing_version": prithvi.preprocessing_version(),
        }, fh, indent=2)

    print("\n[m4b] OOF predictions:", len(oof))
    print(summary[["metric", "n_events", "mean", "median", "bootstrap_ci_lo",
                   "bootstrap_ci_hi"]].to_string(index=False))

    # ---- paired comparison vs frozen PRIMARY RF ----
    rf = pd.read_csv(os.path.join(RESULTS_ROOT, "m2", "event",
                                  "m2_event_oof_predictions.csv"))
    merged = rf.merge(oof, on="event_id", suffixes=("_rf", "_prithvi"))
    paired_rows = []
    for m, rf_c, pr_c in [("iou", "iou_rf", "iou_prithvi"),
                          ("dice", "dice_rf", "dice_prithvi"),
                          ("auprc", "auprc_rf", "auprc_prithvi"),
                          ("auroc", "auroc_rf", "auroc_prithvi")]:
        d = (merged[pr_c] - merged[rf_c]).to_numpy(float)
        lo, hi = bootstrap_ci(d, seed=42)
        paired_rows.append({"metric": m,
                            "rf_mean": float(merged[rf_c].mean()),
                            "prithvi_mean": float(merged[pr_c].mean()),
                            "delta_mean": float(np.nanmean(d)),
                            "delta_median": float(np.nanmedian(d)),
                            "delta_ci_lo": lo, "delta_ci_hi": hi})
    pd.DataFrame(paired_rows).to_csv(
        os.path.join(OUT_ROOT, "m4_paired_vs_rf.csv"), index=False)
    print("\n[m4b] paired Prithvi-linear vs RF:")
    for r in paired_rows:
        print(f"  {r['metric']:6s} rf={r['rf_mean']:.4f} "
              f"prithvi={r['prithvi_mean']:.4f} delta={r['delta_mean']:+.4f} "
              f"CI=[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]")
    print(f"outputs -> {OUT_ROOT}")


if __name__ == "__main__":
    main()

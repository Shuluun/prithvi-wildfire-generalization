"""M4.5 diagnostic — per-layer linear-probe controls (representation layers).

Answers "which depth carries the strongest burn-scar information?" using ONLY the
already-cached frozen representations. Trains a single 1x1 conv probe on ONE frozen
layer (1024 channels) at a time — layer 5, 11, 17, 23 — under the IDENTICAL folds,
supervised pixel budget (<=2048 px/chip), inner threshold policy (maximize burned
Dice on a 99-point grid), and optimization budget (AdamW lr=1e-3, wd=1e-4, max 50
epochs, patience 5, BCE + pos_weight) as the M4b multi-layer concatenation probe.

This is a diagnostic control only: no layer-combination search, no decoder, no
fine-tuning, no outer-test tuning.

Usage (from repo root):
  python scripts/m4b_layer_probe.py --layers 5,11,17,23
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
from src.evaluation.bootstrap import summarize_events  # noqa: E402
from src.evaluation.metrics import per_chip_metrics, select_threshold  # noqa: E402
from src.features.spectral import chip_rng, sample_valid_indices  # noqa: E402

SEED = 42
CAP = 2048
THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FEAT = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache", "features")
OUT_ROOT = os.path.join(RESULTS_ROOT, "m4_5_layers")

METRIC_COLS = ["iou", "dice", "precision", "recall", "auprc", "auroc"]
OPT = dict(lr=1e-3, weight_decay=1e-4, max_epochs=50, patience=5)

# cache axis-0 index -> Prithvi layer number (see m4a_prithvi_feasibility.py)
LAYER_TO_IDX = {5: 0, 11: 1, 17: 2, 23: 3}


class LinearProbe(nn.Module):
    def __init__(self, in_ch=1024):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, feat32):
        logits = self.conv(feat32)
        return F.interpolate(logits, size=(512, 512), mode="bilinear",
                             align_corners=False)

    @property
    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _mask_path(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir,
                        image_id + "_merged.tif").replace("_merged.tif", ".mask.tif")


def load_mask(image_id, split_dir):
    import rasterio
    with rasterio.open(_mask_path(image_id, split_dir)) as ds:
        return ds.read(1).astype(np.int8)


def make_load_features(layer):
    idx = LAYER_TO_IDX[layer]

    def load_features(image_id):
        a = np.load(os.path.join(CACHE_FEAT, f"{image_id}.npy"))  # (4,1024,32,32)
        a = a[idx].astype(np.float32)[None]  # (1,1024,32,32)
        return torch.from_numpy(a).to(DEVICE)
    return load_features


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def eval_chip_logits(probe, image_id, split_dir, load_features):
    mask = load_mask(image_id, split_dir)
    valid = mask >= 0
    logits = probe(load_features(image_id))[0, 0]
    yv = mask[valid].astype(np.int64)
    lv = logits[valid].detach().cpu().numpy().astype(np.float32)
    return lv, yv


def sampled_pixel_logits_labels(probe, image_id, split_dir, rng, load_features, cap=CAP):
    mask = load_mask(image_id, split_dir)
    idx = sample_valid_indices(mask.ravel(), cap, rng)
    rows, cols = idx // mask.shape[1], idx % mask.shape[1]
    y = mask.ravel()[idx].astype(np.int64)
    logits = probe(load_features(image_id))
    return logits[0, 0, rows, cols], torch.from_numpy(y).to(DEVICE)


def fold_val_dice(probe, val_rows, img_split, load_features):
    yt, yp = [], []
    for image_id in val_rows["image_id"]:
        lv, yv = eval_chip_logits(probe, image_id, img_split[image_id],
                                  load_features)
        if len(yv) == 0:
            continue
        yt.append(yv); yp.append(lv)
    if not yt:
        return 0.0, None
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return select_threshold(yt, _sigmoid(yp), THRESHOLD_GRID)


def train_probe(train_rows, val_rows, img_split, pos_weight, load_features):
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
                probe, image_id, img_split[image_id], rng, load_features, CAP)
            if len(y) == 0:
                continue
            loss = loss_fn(logits_s, y.float())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")

        val_dice, _ = fold_val_dice(probe, val_rows, img_split, load_features)
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


def run_layer(layer):
    load_features = make_load_features(layer)
    out_dir = os.path.join(OUT_ROOT, f"layer{layer}")
    os.makedirs(out_dir, exist_ok=True)

    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    assert set(inner["event_id"]) == set(primary["event_id"])

    oof_all, fold_rows = [], []
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        train_rows = grp[grp["role"] == "train"]
        val_rows = grp[grp["role"] == "val"]
        test_rows = grp[grp["role"] == "test"]

        n_pos = n_neg = 0
        for image_id in train_rows["image_id"]:
            m = load_mask(image_id, img_split[image_id])
            n_pos += int((m == 1).sum()); n_neg += int((m == 0).sum())
        pos_weight = n_neg / n_pos if n_pos else 1.0

        probe, best = train_probe(train_rows, val_rows, img_split,
                                  pos_weight, load_features)
        val_dice, thr = fold_val_dice(probe, val_rows, img_split, load_features)

        for image_id, event_id in zip(test_rows["image_id"],
                                      test_rows["event_id"]):
            lv, yv = eval_chip_logits(probe, image_id, img_split[image_id],
                                      load_features)
            yp = _sigmoid(lv)
            m = per_chip_metrics(yv, yp, thr)
            oof_all.append({"event_id": event_id, "image_id": image_id,
                            "fold": f, "threshold": thr, **m})

        fold_rows.append({"fold": f, "threshold": thr, "inner_val_dice": val_dice,
                          "best_epoch": best["epoch"], "train_loss": best["loss"]})
        print(f"[layer{layer}] fold {f}: thr={thr:.4f} val_dice={val_dice:.4f} "
              f"best_epoch={best['epoch']} loss={best['loss']:.4f}")

    oof = pd.DataFrame(oof_all)
    assert oof["event_id"].is_unique and len(oof) == len(primary)
    oof = oof.sort_values("event_id").reset_index(drop=True)
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out_dir, "fold_summary.csv"),
                                   index=False)
    summary = summarize_events(oof, METRIC_COLS)
    summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"layer": layer, "in_channels": 1024, "seed": SEED,
                   "cap_pixels_per_train_chip": CAP, "opt": OPT,
                   "n_primary_events": int(len(primary))}, fh, indent=2)
    print(f"[layer{layer}] done. IoU mean={summary.loc[summary.metric=='iou','mean'].iloc[0]:.4f} "
          f"Dice mean={summary.loc[summary.metric=='dice','mean'].iloc[0]:.4f} "
          f"AUROC mean={summary.loc[summary.metric=='auroc','mean'].iloc[0]:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,11,17,23",
                    help="comma-separated Prithvi layer numbers")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(OUT_ROOT, exist_ok=True)
    for layer in layers:
        run_layer(layer)


if __name__ == "__main__":
    main()

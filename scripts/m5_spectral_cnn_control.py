"""M5 — matched-capacity SPECTRAL spatial control (8 spectral features).

A deliberately small convolutional segmentation model over the SAME 8 spectral
features used by M2 (6 HLS bands + NDVI + NBR), with local 3x3 context and
~0.52M trainable parameters (comparable order to the Prithvi SpatialDecoder's
~0.66M). This is NOT an optimized spectral model: it is a matched-capacity
control answering "does Prithvi add value beyond what a lightweight spatial
model can extract directly from spectral inputs?"

Protocol identical to M2/M4/M5a/M5b: frozen 576-event Wildfire population,
frozen event-disjoint inner K5, <=2048 valid px/train chip, inner-val burned-Dice
threshold, outer test once per fold. No extra engineered features, no dNBR, no
augmentation/post-processing. Nodata (-1) input pixels are zeroed before the
convs (a minimal handling so arbitrary nodata reflectance does not inject into
neighboring valid pixels).

Usage (from repo root):
  python scripts/m5_spectral_cnn_control.py            # full PRIMARY K5
  python scripts/m5_spectral_cnn_control.py --smoke    # NON-SCIENTIFIC dry run
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, RESULTS_ROOT  # noqa: E402
from _m5_common import DEVICE, load_mask, run_experiment  # noqa: E402
from src.m5.models import SpectralCNN  # noqa: E402

OUT_ROOT = os.path.join(RESULTS_ROOT, "m5_spectral_cnn")

ARCH_DESC = ("spectral CNN: 8ch -> Conv3x3 8->32 -> 4x stride-2 downsample to 32x32 "
             "-> 4x bilinear up w/ 3x3 refine -> 1x1 logit; ~0.52M params; no skips")


def merged_path_of(image_id, split_dir):
    return os.path.join(HLS_BURN_SCARS_DIR, split_dir, image_id + "_merged.tif")


def load_spectral_cnn(image_id, split_dir):
    """8-channel spectral features (6 bands + NDVI + NBR) at 512x512, nodata zeroed.

    Band order matches src/features/spectral.py: B02,B03,B04,B8A,B11,B12,NDVI,NBR.
    """
    import rasterio
    with rasterio.open(merged_path_of(image_id, split_dir)) as ds:
        img = ds.read().astype(np.float32)  # (6, 512, 512)
    blue, green, red = img[0], img[1], img[2]
    nir, swir1, swir2 = img[3], img[4], img[5]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)
    x = np.stack([blue, green, red, nir, swir1, swir2, ndvi, nbr], axis=0)
    mask = load_mask(image_id, split_dir)
    valid = (mask >= 0).astype(np.float32)  # zero out nodata reflectance
    x = x * valid[None]
    return torch.from_numpy(np.ascontiguousarray(x))[None].to(DEVICE)


def build_model():
    return SpectralCNN()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    run_experiment(build_model, load_spectral_cnn, OUT_ROOT, "m5_spectral",
                   ARCH_DESC)


def smoke():
    """NON-SCIENTIFIC implementation test (forward + feature ranges + 3-epoch train)."""
    import pandas as pd
    from src.data.paths import METADATA_ROOT
    from src.data.splits import SPLITS_DIR
    from src.features.spectral import FEATURE_NAMES
    from _m5_common import (CAP, OPT, sampled_pixel_logits_labels, train_model)

    m = SpectralCNN().to(DEVICE)
    print("== M5 spectral CNN NON-SCIENTIFIC SMOKE TEST ==")
    print("SpectralCNN param table:", m.param_table())
    print("n_trainable:", m.n_trainable)
    x = torch.zeros(1, 8, 512, 512, device=DEVICE)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512), y.shape
    print("forward (1,8,512,512) ->", tuple(y.shape), "OK")

    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    grp = inner[inner["outer_fold"] == 1]
    train_rows = grp[grp["role"] == "train"].head(8)
    val_rows = grp[grp["role"] == "val"].head(6)

    # feature sanity on one chip
    f = load_spectral_cnn(train_rows["image_id"].iloc[0],
                          img_split[train_rows["image_id"].iloc[0]])
    print("feature tensor shape", tuple(f.shape), "names", FEATURE_NAMES)
    print("feature channel means", f[0].mean(dim=(1, 2)).tolist())

    n_pos = n_neg = 0
    for image_id in train_rows["image_id"]:
        mm = load_mask(image_id, img_split[image_id])
        n_pos += int((mm == 1).sum()); n_neg += int((mm == 0).sum())
    pos_weight = n_neg / n_pos if n_pos else 1.0

    model = SpectralCNN().to(DEVICE)
    rng = np.random.default_rng(0)
    ls, yv = sampled_pixel_logits_labels(model, load_spectral_cnn,
                                         train_rows["image_id"].iloc[0],
                                         img_split[train_rows["image_id"].iloc[0]],
                                         rng, CAP)
    print("sampled logits shape", tuple(ls.shape), "labels", tuple(yv.shape))
    model, best = train_model(model, load_spectral_cnn, train_rows, val_rows,
                              img_split, pos_weight, opt=dict(OPT, max_epochs=3))
    print("smoke train done: best_epoch", best["epoch"],
          "val_dice", round(best["val_dice"], 4))
    print("M5 spectral CNN smoke: PASS (NON-SCIENTIFIC)")


if __name__ == "__main__":
    main()

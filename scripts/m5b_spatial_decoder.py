"""M5b — lightweight SPATIAL decoder on frozen Prithvi features.

Hypothesis test (M5 gate B): does adding LOCAL SPATIAL CONTEXT (3x3 convs +
progressive bilinear upsampling) over the frozen layers [5,11,17,23] recover a
cross-event burn-scar direction that the pointwise readouts could not?

Model: per-layer 1x1 1024->64 -> concat 256 -> Conv3x3 256->128 (GELU) ->
Conv3x3 128->64 (GELU) -> progressive bilinear 2x upsampling with small 3x3
conv refinement -> 1-channel logit at 512. ~0.66M trainable params (<= 1-2M).
No transformer/attention/UNet/UperNet/CRF/morphology/TTA/ensemble; encoder frozen.

Protocol identical to M2/M4/M5a (frozen 576-event Wildfire population, frozen
event-disjoint inner K5, <=2048 valid px/train chip, inner-val burned-Dice
threshold, outer test once per fold).

Usage (from repo root):
  python scripts/m5b_spatial_decoder.py            # full PRIMARY K5
  python scripts/m5b_spatial_decoder.py --smoke    # NON-SCIENTIFIC dry run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _m5_common import (  # noqa: E402
    RESULTS_ROOT, load_prithvi_layers, run_experiment,
)
from src.m5.models import SpatialDecoder  # noqa: E402

OUT_ROOT = os.path.join(RESULTS_ROOT, "m5b_spatial_decoder")

ARCH_DESC = ("per-layer 1x1 1024->64 -> concat 256 -> Conv3x3 256->128 GELU -> "
             "Conv3x3 128->64 GELU -> bilinear x2 (x4) w/ 3x3 refine (64->32->16->8) "
             "-> 1x1 logit; ~0.66M params")


def build_model():
    return SpatialDecoder()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    run_experiment(build_model, load_prithvi_layers, OUT_ROOT, "m5b", ARCH_DESC)


def smoke():
    """NON-SCIENTIFIC implementation test (forward shapes + param count + 3-epoch train)."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.paths import METADATA_ROOT
    from src.data.splits import SPLITS_DIR
    from _m5_common import (CAP, DEVICE, OPT, load_mask, load_prithvi_layers,
                            sampled_pixel_logits_labels, train_model)

    m = SpatialDecoder().to(DEVICE)
    print("== M5b NON-SCIENTIFIC SMOKE TEST ==")
    print("SpatialDecoder param table:", m.param_table())
    print("n_trainable:", m.n_trainable)
    assert m.n_trainable <= 2_000_000, "decoder exceeds 2M param budget"
    x = torch.zeros(1, 4, 1024, 32, 32, device=DEVICE)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512), y.shape
    print("forward (1,4,1024,32,32) ->", tuple(y.shape), "OK")

    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    grp = inner[inner["outer_fold"] == 1]
    train_rows = grp[grp["role"] == "train"].head(8)
    val_rows = grp[grp["role"] == "val"].head(6)

    n_pos = n_neg = 0
    for image_id in train_rows["image_id"]:
        mm = load_mask(image_id, img_split[image_id])
        n_pos += int((mm == 1).sum()); n_neg += int((mm == 0).sum())
    pos_weight = n_neg / n_pos if n_pos else 1.0

    model = SpatialDecoder().to(DEVICE)
    rng = np.random.default_rng(0)
    ls, yv = sampled_pixel_logits_labels(model, load_prithvi_layers,
                                         train_rows["image_id"].iloc[0],
                                         img_split[train_rows["image_id"].iloc[0]],
                                         rng, CAP)
    print("sampled logits shape", tuple(ls.shape), "labels", tuple(yv.shape))
    model, best = train_model(model, load_prithvi_layers, train_rows, val_rows,
                              img_split, pos_weight, opt=dict(OPT, max_epochs=3))
    print("smoke train done: best_epoch", best["epoch"],
          "val_dice", round(best["val_dice"], 4))
    print("M5b smoke: PASS (NON-SCIENTIFIC)")


if __name__ == "__main__":
    main()

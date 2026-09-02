"""M5a — nonlinear POINTWISE probe on frozen Prithvi features.

Hypothesis test (M5 gate A): is burned-area information *nonlinearly* readable
from the frozen concatenated layers [5,11,17,23] without any spatial context?

Model: 4096 -> Linear(256) -> GELU -> Dropout(0.1) -> Linear(64) -> GELU ->
Linear(1), applied independently at every 32x32 token, then bilinear to 512.
No 3x3 conv, no spatial attention, no learned upsampling, no skip connections.

Protocol: identical to M2/M4b (frozen 576-event Wildfire population, frozen
event-disjoint inner K5, <=2048 valid px/train chip, inner-val burned-Dice
threshold, outer test touched once per fold). Produces event-level OOF metrics
and paired delta bootstrap CIs vs the RF baseline and the M4 linear probe.

Usage (from repo root):
  python scripts/m5a_pointwise_mlp.py            # full PRIMARY K5
  python scripts/m5a_pointwise_mlp.py --smoke    # NON-SCIENTIFIC dry run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _m5_common import (  # noqa: E402
    DEVICE, RESULTS_ROOT, load_prithvi_concat, paired_delta, run_experiment,
)
from src.m5.models import PointwiseMLP  # noqa: E402

OUT_ROOT = os.path.join(RESULTS_ROOT, "m5a_pointwise_mlp")
RF_OOF = os.path.join(RESULTS_ROOT, "m2", "event", "m2_event_oof_predictions.csv")
LIN_OOF = os.path.join(RESULTS_ROOT, "m4_linear_probe",
                       "m4_event_oof_predictions.csv")

ARCH_DESC = ("pointwise MLP 4096->256(GELU+Dropout0.1)->64(GELU)->1 (1x1 convs) "
             "+ bilinear 32->512; no spatial context")


def build_model():
    return PointwiseMLP()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    oof, _summary = run_experiment(build_model, load_prithvi_concat, OUT_ROOT,
                                    "m5a", ARCH_DESC)

    # paired deltas: MLP - linear, MLP - RF
    print("\n[m5a] paired MLP - linear:")
    d_lin = paired_delta(oof, LIN_OOF, "linear", "mlp",
                         out_path=os.path.join(OUT_ROOT, "m5a_paired_vs_linear.csv"))
    for _, r in d_lin.iterrows():
        print(f"  {r['metric']:6s} lin={r['linear_mean']:.4f} "
              f"mlp={r['mlp_mean']:.4f} delta={r['delta_mean']:+.4f} "
              f"CI=[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]")

    print("\n[m5a] paired MLP - RF:")
    d_rf = paired_delta(oof, RF_OOF, "rf", "mlp",
                        out_path=os.path.join(OUT_ROOT, "m5a_paired_vs_rf.csv"))
    for _, r in d_rf.iterrows():
        print(f"  {r['metric']:6s} rf={r['rf_mean']:.4f} "
              f"mlp={r['mlp_mean']:.4f} delta={r['delta_mean']:+.4f} "
              f"CI=[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]")


def smoke():
    """NON-SCIENTIFIC implementation test on fold 1, a few epochs, tiny subset."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.paths import METADATA_ROOT
    from src.data.splits import SPLITS_DIR
    from _m5_common import (CAP, SEED, OPT, load_mask, sampled_pixel_logits_labels,
                            eval_chip_logits, fold_val_dice, train_model)

    m = PointwiseMLP().to(DEVICE)
    print("== M5a NON-SCIENTIFIC SMOKE TEST ==")
    print("PointwiseMLP param table:", m.param_table())
    print("n_trainable:", m.n_trainable)
    x = torch.zeros(1, 4096, 32, 32, device=DEVICE)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512), y.shape
    print("forward (1,4096,32,32) ->", tuple(y.shape), "OK")

    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))
    inner = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_inner_seed42.csv"))
    grp = inner[inner["outer_fold"] == 1]
    train_rows = grp[grp["role"] == "train"].head(8)
    val_rows = grp[grp["role"] == "val"].head(6)
    test_rows = grp[grp["role"] == "test"].head(6)

    n_pos = n_neg = 0
    for image_id in train_rows["image_id"]:
        mm = load_mask(image_id, img_split[image_id])
        n_pos += int((mm == 1).sum()); n_neg += int((mm == 0).sum())
    pos_weight = n_neg / n_pos if n_pos else 1.0

    model = PointwiseMLP().to(DEVICE)
    rng = np.random.default_rng(0)
    ls, yv = sampled_pixel_logits_labels(model, load_prithvi_concat,
                                         train_rows["image_id"].iloc[0],
                                         img_split[train_rows["image_id"].iloc[0]],
                                         rng, CAP)
    print("sampled logits shape", tuple(ls.shape), "labels", tuple(yv.shape))
    model, best = train_model(model, load_prithvi_concat, train_rows, val_rows,
                              img_split, pos_weight,
                              opt=dict(OPT, max_epochs=3))
    print("smoke train done: best_epoch", best["epoch"],
          "val_dice", round(best["val_dice"], 4))
    for image_id, event_id in zip(test_rows["image_id"], test_rows["event_id"]):
        lv, yvv = eval_chip_logits(model, load_prithvi_concat, image_id,
                                   img_split[image_id])
        print(f"  test {event_id[:20]} n_valid={len(yvv)} "
              f"pos_frac={yvv.mean():.3f}")
    print("M5a smoke: PASS (NON-SCIENTIFIC)")


if __name__ == "__main__":
    main()

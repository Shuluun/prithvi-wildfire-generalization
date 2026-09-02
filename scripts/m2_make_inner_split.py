"""M2 step 1: freeze the event-disjoint INNER train/val protocol.

For each of the three M1.6 outer 5-fold manifests (event / spatial /
tiledisjoint) this builds, saves, and audits the inner train/val split of
every outer training pool. The inner split is deterministic (seed 42) and is
the SAME assignment both the spectral-RF baseline (M2/M3) and the frozen
Prithvi decoder (M5) will use for threshold/model selection and early stopping.

PROTOCOL CORRECTION / FREEZE ONLY — no model training, no pixel touching.

Usage (from repo root):
  python scripts/m2_make_inner_split.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT  # noqa: E402
from src.data.splits import SPLITS_DIR, get_inner_split  # noqa: E402

SEED = 42
VAL_FRAC = 0.2
K = 5

PROTOCOLS = ["event", "spatial", "tiledisjoint"]
METRIC_COLS = ["iou", "dice", "precision", "recall", "auprc", "auroc"]


def load_outer_manifest(protocol):
    return pd.read_csv(os.path.join(
        SPLITS_DIR, f"split_k5_{protocol}_seed42.csv"))


def outer_fold_of(manifest):
    test = manifest[manifest["role"] == "test"]
    return dict(zip(test["event_id"], test["fold"]))


def main():
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    primary_rows = primary[["event_id", "image_id", "fire_year",
                            "burned_fraction"]]
    print(f"primary population: {len(primary_rows)} events (Wildfire)")
    assert primary_rows["event_id"].is_unique
    assert primary_rows["image_id"].is_unique

    for protocol in PROTOCOLS:
        outer = load_outer_manifest(protocol)
        fold_of = outer_fold_of(outer)
        inner = get_inner_split(primary_rows, fold_of,
                                val_frac=VAL_FRAC, seed=SEED)
        # determinism check
        inner2 = get_inner_split(primary_rows, fold_of,
                                 val_frac=VAL_FRAC, seed=SEED)
        pd.testing.assert_frame_equal(inner, inner2)
        out = os.path.join(SPLITS_DIR, f"split_k5_{protocol}_inner_seed42.csv")
        inner.to_csv(out, index=False)
        print(f"\n=== {protocol} inner manifest -> {out} ===")
        _audit(inner, primary)


def _audit(inner, primary):
    primary = primary.set_index("event_id")
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        counts = grp["role"].value_counts()
        n_test = int(counts.get("test", 0))
        n_train = int(counts.get("train", 0))
        n_val = int(counts.get("val", 0))
        print(f"  fold {f}: train={n_train} val={n_val} test={n_test} "
              f"(inner train/val ratio {n_val}/{n_train + n_val:.3f})")

    # disjointness: per outer fold, roles are mutually exclusive
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        assert not grp.duplicated(["event_id", "outer_fold"]).any()
        for role in ("train", "val", "test"):
            ids = set(grp[grp["role"] == role]["event_id"])
            other = set(grp[grp["role"] != role]["event_id"])
            assert not (ids & other), f"fold {f} role {role} overlaps"
    # no event in both inner-train and inner-val within the same fold
    for f in sorted(inner["outer_fold"].unique()):
        grp = inner[inner["outer_fold"] == f]
        tv = set(grp[grp["role"] == "train"]["event_id"])
        vv = set(grp[grp["role"] == "val"]["event_id"])
        assert not (tv & vv), f"fold {f} inner train/val overlap"
    # every event tested exactly once across folds
    per_event = (inner.groupby("event_id")["role"].value_counts()
                 .unstack(fill_value=0))
    assert (per_event["test"] == 1).all(), "an event must be tested once"

    # distribution summaries: year + burned fraction for inner-val vs all
    val_ids = set(inner[inner["role"] == "val"]["event_id"])
    train_ids = set(inner[inner["role"] == "train"]["event_id"])
    val_fracs = primary.loc[list(val_ids & set(primary.index)),
                            "burned_fraction"]
    tr_fracs = primary.loc[list(train_ids & set(primary.index)),
                           "burned_fraction"]
    print(f"  inner-val events: {len(val_ids)} | burned-fraction "
          f"mean {val_fracs.mean():.4f} vs train {tr_fracs.mean():.4f} "
          f"(pooled primary {primary['burned_fraction'].mean():.4f})")


if __name__ == "__main__":
    main()

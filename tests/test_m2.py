"""Spectral-baseline tests.

Covers the frozen M2 protocol pieces that are testable without running the
full experiment: deterministic pixel sampling, no-data exclusion, feature
dimension and NDVI/NBR finiteness, outer train/val/test event disjointness,
one-and-only-one OOF prediction per primary event, spatial zero-overlap, and
metrics-on-valid-pixels-only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.data.splits import (  # noqa: E402
    SPLITS_DIR, assert_no_cross_fold_overlap, get_inner_split,
)
from src.evaluation.metrics import per_chip_metrics, select_threshold  # noqa: E402
from src.features.spectral import (  # noqa: E402
    FEATURE_NAMES, chip_rng, load_features, sample_valid_indices,
    valid_features_labels,
)

ATTRS = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
PRIMARY = ATTRS[ATTRS["incid_type"] == "Wildfire"]
IMG_SPLIT = dict(zip(PRIMARY["image_id"], PRIMARY["split"]))


# --- deterministic pixel sampling -------------------------------------------

def test_feature_dimension():
    assert len(FEATURE_NAMES) == 8
    assert FEATURE_NAMES == ["B02", "B03", "B04", "B8A", "B11", "B12",
                             "NDVI", "NBR"]


def test_sampling_deterministic():
    mask = np.array([0, 1, 1, -1, 0, 1, 0, 0, 1, -1])
    a = sample_valid_indices(mask, 5, chip_rng(42, "chip_x"))
    b = sample_valid_indices(mask, 5, chip_rng(42, "chip_x"))
    assert np.array_equal(a, b)


def test_sampling_excludes_nodata():
    mask = np.array([-1, -1, 0, 1, 0, 1, -1])
    idx = sample_valid_indices(mask, 1000, np.random.default_rng(0))
    assert -1 not in set(mask[idx])
    assert set(mask[idx]) <= {0, 1}


def test_sampling_subset_no_replacement_cap():
    mask = np.array([0] * 90 + [1] * 10)
    idx = sample_valid_indices(mask, 20, np.random.default_rng(0))
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20  # without replacement
    assert set(mask[idx]) <= {0, 1}
    # fewer than cap -> use all valid pixels
    idx_all = sample_valid_indices(mask, 1000, np.random.default_rng(0))
    assert len(idx_all) == 100


def test_valid_features_labels_excludes_nodata():
    x = np.arange(8).reshape(4, 2).astype(float)
    mask = np.array([-1, 0, 1, 0])
    xv, yv = valid_features_labels(x, mask)
    assert np.array_equal(yv, np.array([0, 1, 0]))
    assert len(xv) == 3


# --- real-chip feature checks (skip if raw data absent) ----------------------

@pytest.fixture(scope="module")
def one_chip():
    image_id = PRIMARY["image_id"].iloc[0]
    mp = os.path.join(HLS_BURN_SCARS_DIR, IMG_SPLIT[image_id],
                      image_id + "_merged.tif")
    if not os.path.exists(mp):
        pytest.skip("raw hls_burn_scars data not present")
    return load_features(mp), image_id


def test_real_chip_feature_dim_and_mask(one_chip):
    (x, mask), _ = one_chip
    assert x.shape[1] == 8
    assert set(np.unique(mask)) <= {-1, 0, 1}


def test_real_chip_ndvi_nbr_finite(one_chip):
    (x, mask), _ = one_chip
    valid = mask >= 0
    ndvi, nbr = x[valid, 6], x[valid, 7]
    assert np.isfinite(ndvi).all()
    assert np.isfinite(nbr).all()
    assert np.all(ndvi >= -1 - 1e-4) and np.all(ndvi <= 1 + 1e-4)


# --- metrics ignore no-data (caller excludes; functions see {0,1}) -----------

def test_per_chip_metrics_on_valid_only():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    m = per_chip_metrics(y_true, y_prob, 0.5)
    assert m["n_valid"] == 5
    for k in ("iou", "dice", "precision", "recall", "auprc", "auroc"):
        assert np.isfinite(m[k]), k
    assert 0.0 <= m["true_burn_fraction"] <= 1.0
    assert 0.0 <= m["pred_burn_fraction"] <= 1.0


def test_select_threshold_in_grid():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.9, 0.4, 0.7])
    grid = np.array([0.2, 0.5, 0.8])
    t, d = select_threshold(y_true, y_prob, grid)
    assert t in grid
    assert 0.0 <= d <= 1.0


# --- frozen inner-manifest disjointness --------------------------------------

@pytest.fixture(scope="module")
def inner():
    return pd.read_csv(os.path.join(SPLITS_DIR,
                                    "split_k5_event_inner_seed42.csv"))


def test_inner_covers_primary_exactly(inner):
    assert set(inner["event_id"]) == set(PRIMARY["event_id"])
    assert set(inner["image_id"]) == set(PRIMARY["image_id"])
    assert set(inner.columns) == {"image_id", "event_id", "outer_fold", "role"}


def test_inner_disjoint_per_fold(inner):
    for f, grp in inner.groupby("outer_fold"):
        assert not grp.duplicated("event_id").any()
        for role in ("train", "val", "test"):
            ids = set(grp[grp["role"] == role]["event_id"])
            other = set(grp[grp["role"] != role]["event_id"])
            assert not (ids & other), f"fold {f} role {role} overlaps"


def test_inner_no_train_val_overlap(inner):
    for f, grp in inner.groupby("outer_fold"):
        tv = set(grp[grp["role"] == "train"]["event_id"])
        vv = set(grp[grp["role"] == "val"]["event_id"])
        assert not (tv & vv)


def test_one_and_only_one_oof_per_event(inner):
    per_event = (inner.groupby("event_id")["role"]
                 .value_counts().unstack(fill_value=0))
    assert (per_event["test"] == 1).all()
    assert (per_event["train"] + per_event["val"] == 4).all()


def test_inner_deterministic():
    outer = pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_seed42.csv"))
    fold_of = dict(zip(outer[outer["role"] == "test"]["event_id"],
                       outer[outer["role"] == "test"]["fold"]))
    rows = PRIMARY[["event_id", "image_id", "fire_year", "burned_fraction"]]
    a = get_inner_split(rows, fold_of)
    b = get_inner_split(rows, fold_of)
    pd.testing.assert_frame_equal(a, b)


def test_spatial_inner_zero_cross_fold_overlap():
    inner_sp = pd.read_csv(os.path.join(
        SPLITS_DIR, "split_k5_spatial_inner_seed42.csv"))
    test = (inner_sp[inner_sp["role"] == "test"]
            .rename(columns={"outer_fold": "fold"})[["image_id", "event_id",
                                                     "fold", "role"]])
    overlap = pd.read_csv(os.path.join(RESULTS_ROOT, "reports",
                                       "event_overlap_pairs_primary.csv"))
    assert assert_no_cross_fold_overlap(test, overlap) == []

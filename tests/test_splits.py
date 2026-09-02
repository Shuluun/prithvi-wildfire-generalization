"""Leakage tests for the deterministic split manifests.

Run: python -m pytest tests/ -q  (from repo root)

Covers the LEGACY manifests (chip-level reference + LOEO, M1.5) and the
M1.6 primary protocol (5-fold event-disjoint CV over the primary analysis
population: matched Wildfire events; shared folds; spatial-clean
sensitivity with a proven zero cross-fold footprint overlap).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.data.splits import (  # noqa: E402
    SPLITS_DIR, assert_no_cross_fold_overlap, get_kfold_event_split,
    get_within_event_split, loeo_manifest,
)

INVENTORY = pd.read_csv(os.path.join(METADATA_ROOT, "chip_inventory.csv"))
EVENTS = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
CHIP_TABLE = INVENTORY[["image_id"]].merge(
    EVENTS[["image_id", "event_id", "match_status"]], on="image_id", how="left")


@pytest.fixture(scope="module")
def within():
    return pd.read_csv(os.path.join(SPLITS_DIR, "split_within_seed42.csv"))


@pytest.fixture(scope="module")
def loeo():
    path = os.path.join(SPLITS_DIR, "split_loeo_seed42.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Regenerable from committed chip_inventory.csv + events.csv (the ~55 MB
    # manifest is gitignored; see scripts/m1_5_make_splits.py). Rebuild + cache.
    manifest = loeo_manifest(CHIP_TABLE, seed=42)
    manifest.to_csv(path, index=False)
    return manifest


def test_inventory_matches_manifest_rows(within, loeo):
    # conventional: every chip appears; LOEO: exactly the matched (event-id
    # bearing) chips appear — ambiguous/unmatched chips are excluded by design
    inv_ids = set(INVENTORY["image_id"])
    usable_ids = set(CHIP_TABLE[CHIP_TABLE["event_id"].notna()]["image_id"])
    assert set(within["image_id"]) == inv_ids
    assert set(loeo["image_id"]) == usable_ids
    assert usable_ids < inv_ids  # exclusion is real, not accidental


def test_within_chip_disjoint(within):
    # a chip belongs to exactly one split
    assert not within["image_id"].duplicated().any()
    counts = within["split"].value_counts()
    assert set(counts.index) == {"train", "val", "test"}
    assert counts.sum() == len(INVENTORY) == 804


def test_no_pixel_level_splitting(within):
    # the manifest assigns whole chips only — there is no row-level (pixel)
    # assignment anywhere; chips are the atomic unit (spec forbids pixel splits)
    assert within["split"].isin(["train", "val", "test"]).all()
    assert len(within) == within["image_id"].nunique()


def test_loeo_event_disjoint_per_fold(loeo):
    # for every held-out event: test chips all belong to it; no train chip
    # belongs to it; no chip appears in two roles
    assert not loeo.duplicated(["image_id", "held_out_event"]).any()
    for held, grp in loeo.groupby("held_out_event"):
        test_rows = grp[grp["role"] == "test"]
        train_rows = grp[grp["role"] == "train"]
        assert len(test_rows) >= 1
        assert (test_rows["event_id"] == held).all()
        assert not (train_rows["event_id"] == held).any()


def test_loeo_uses_only_matched_events(loeo):
    # LOEO folds exist only for matched events (ambiguous/unmatched rows have
    # no event_id and are excluded rather than forced)
    usable = CHIP_TABLE[CHIP_TABLE["event_id"].notna()]
    assert set(loeo["held_out_event"]) == set(usable["event_id"])
    assert loeo["event_id"].notna().all()


def test_no_duplicate_image_leakage(within, loeo):
    # no duplicated scenes smuggled in under another name, and train/test
    # never share a chip in any fold
    inv_ids = set(INVENTORY["image_id"])
    assert set(within["image_id"]) == inv_ids
    assert set(loeo["image_id"]).issubset(inv_ids)
    for held, grp in loeo.groupby("held_out_event"):
        t = set(grp[grp["role"] == "test"]["image_id"])
        tr = set(grp[grp["role"] == "train"]["image_id"])
        assert not (t & tr)


def test_manifests_deterministic():
    # regenerating with the same seed reproduces the same assignment
    w1 = get_within_event_split(CHIP_TABLE, seed=42)
    w2 = pd.read_csv(os.path.join(SPLITS_DIR, "split_within_seed42.csv"))
    pd.testing.assert_frame_equal(
        w1.sort_values("image_id").reset_index(drop=True),
        w2.sort_values("image_id").reset_index(drop=True))


def test_chip_level_split_is_automatically_event_disjoint(within):
    # M1.6 structural finding: every event has exactly 1 chip, so the
    # chip-level reference split CANNOT be a within-event protocol — no event
    # contributes chips to two roles. The legacy manifest is event-disjoint by
    # construction, not by design intent; it is NOT the primary population.
    wm = within.merge(EVENTS[["image_id", "event_id"]], on="image_id",
                      how="left")
    n_multi = (wm.groupby("event_id")["split"].nunique() > 1).sum()
    assert n_multi == 0


@pytest.mark.parametrize("seed", [0, 42, 2024])
def test_seed_reproducibility(seed):
    a = get_within_event_split(CHIP_TABLE, seed=seed)
    b = get_within_event_split(CHIP_TABLE, seed=seed)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# M1.6: primary protocol — 5-fold event-disjoint CV (shared folds)
# ---------------------------------------------------------------------------

ATTRS = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
PRIMARY_ATTRS = ATTRS[ATTRS["incid_type"] == "Wildfire"]
PRIMARY_EVENTS = set(PRIMARY_ATTRS["event_id"])
PRIMARY_IMAGES = set(PRIMARY_ATTRS["image_id"])
OVERLAP_PRIMARY = pd.read_csv(os.path.join(
    RESULTS_ROOT, "reports", "event_overlap_pairs_primary.csv"))
K = 5


@pytest.fixture(scope="module")
def k5_event():
    return pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_event_seed42.csv"))


@pytest.fixture(scope="module")
def k5_spatial():
    return pd.read_csv(os.path.join(SPLITS_DIR, "split_k5_spatial_seed42.csv"))


@pytest.fixture(scope="module")
def k5_tile():
    return pd.read_csv(os.path.join(SPLITS_DIR,
                                    "split_k5_tiledisjoint_seed42.csv"))


def test_k5_primary_population(k5_event, k5_spatial, k5_tile):
    # every manifest covers EXACTLY the primary analysis population
    # (matched Wildfire events) — no prescribed/unknown/ambiguous/unmatched
    for m in (k5_event, k5_spatial, k5_tile):
        assert set(m["event_id"]) == PRIMARY_EVENTS
        assert set(m["image_id"]) == PRIMARY_IMAGES
        assert set(m.columns) == {"image_id", "event_id", "fold", "role"}


def test_k5_event_disjoint_per_fold(k5_event):
    # per fold: test = that fold's events; train = all other primary events;
    # no event appears in two roles within one fold
    assert not k5_event.duplicated(["image_id", "fold"]).any()
    for fold, grp in k5_event.groupby("fold"):
        test_rows = grp[grp["role"] == "test"]
        train_rows = grp[grp["role"] == "train"]
        assert len(test_rows) >= 1
        assert len(test_rows) + len(train_rows) == len(PRIMARY_EVENTS)
        assert not (set(test_rows["event_id"]) & set(train_rows["event_id"]))
    # every event tested exactly once, trained exactly K-1 times
    per_event = (k5_event.groupby("event_id")["role"]
                 .value_counts().unstack(fill_value=0))
    assert (per_event["test"] == 1).all()
    assert (per_event["train"] == K - 1).all()


def test_k5_single_shared_assignment(k5_event):
    # one fold assignment per event, model-agnostic (no model-specific folds;
    # no model column in the manifest)
    tested = k5_event[k5_event["role"] == "test"]
    assert tested["event_id"].is_unique
    assert tested["fold"].nunique() == K


def test_k5_spatial_zero_cross_fold_overlap(k5_spatial):
    # the spatial-clean sensitivity: no overlapping pair ever crosses the
    # train/test boundary (checked against the recomputed primary pairs and
    # against the M1.5 audit CSV restricted to primary events)
    assert assert_no_cross_fold_overlap(k5_spatial, OVERLAP_PRIMARY) == []
    all_audit = pd.read_csv(os.path.join(RESULTS_ROOT, "reports",
                                         "event_overlap_pairs.csv"))
    restricted = all_audit[all_audit["event_a"].isin(PRIMARY_EVENTS)
                           & all_audit["event_b"].isin(PRIMARY_EVENTS)]
    assert assert_no_cross_fold_overlap(k5_spatial, restricted) == []


def test_k5_spatial_components_share_fold(k5_spatial):
    # stronger property: every overlapping event pair shares ONE fold
    fold_of = (k5_spatial[k5_spatial["role"] == "test"]
               .drop_duplicates("event_id")
               .set_index("event_id")["fold"].to_dict())
    for _, r in OVERLAP_PRIMARY.iterrows():
        assert fold_of[r["event_a"]] == fold_of[r["event_b"]]


def test_k5_random_manifest_has_cross_fold_pairs(k5_event):
    # the (non-spatial) random K5 DOES have overlapping train/test pairs —
    # this contrast is exactly what the spatial sensitivity quantifies
    # (deterministic property of the recorded manifests)
    violations = assert_no_cross_fold_overlap(k5_event, OVERLAP_PRIMARY)
    assert len(violations) >= 1


def test_k5_tile_disjoint(k5_tile):
    # stricter sensitivity: events sharing an HLS tile share a fold
    fold_of = (k5_tile[k5_tile["role"] == "test"]
               .drop_duplicates("event_id")
               .set_index("event_id")["fold"].to_dict())
    tiles_folds = PRIMARY_ATTRS[["event_id", "hls_tile"]].copy()
    tiles_folds["fold"] = tiles_folds["event_id"].map(fold_of)
    n_tiles_multi = (tiles_folds.groupby("hls_tile")["fold"].nunique() > 1)
    assert not n_tiles_multi.any()


def test_k5_manifest_deterministic(k5_event):
    # regenerating with the same procedure + seed reproduces the PRIMARY
    # manifest exactly
    rows = PRIMARY_ATTRS[["event_id", "image_id", "fire_year",
                          "burned_fraction"]]
    regen, _ = get_kfold_event_split(rows, k=K, seed=42)
    pd.testing.assert_frame_equal(k5_event, regen)

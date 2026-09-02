"""Deterministic split utilities (event-disjoint cross-validation protocols).

Split hierarchy is EVENT -> CHIP -> PIXEL. All splits happen at the
event/chip level before any pixel is touched. Pixel-level random splitting
is forbidden by the project spec and by the leakage tests.

M1.6 protocol correction (2026-08-31): hls_burn_scars has exactly ONE chip
per matched event, so a leak-free "within-event" chip-level protocol does
not exist for this dataset — any chip-level holdout is automatically
event-disjoint (see the M1.6 report). The PRIMARY protocol is deterministic
event-disjoint K-fold CV over the primary analysis population (matched
Wildfire events), with ONE shared fold assignment used by both the
spectral-RF baseline and the frozen-Prithvi model.

Protocols:
- get_kfold_event_split: PRIMARY 5-fold event-disjoint CV, stratified by
  fire_year x burned-fraction quantile bin (seed 42).
- get_spatial_grouped_kfold: sensitivity — connected components of the
  event-footprint overlap graph each stay inside one fold; zero train/test
  footprint overlap (proven with assert_no_cross_fold_overlap).
- get_tile_grouped_kfold: stricter sensitivity (NOT primary) — events
  sharing an HLS tile stay inside one fold.
- LEGACY artifacts (kept for reference, NOT primary): chip-level reference
  split (get_within_event_split) and LOEO (loeo_manifest).

All splits are deterministic given (seed, manifest inputs). Manifests are
written to data/metadata/splits/ and re-checked by tests.
"""
import os

import numpy as np
import pandas as pd

from src.data.paths import METADATA_ROOT

SPLITS_DIR = os.path.join(METADATA_ROOT, "splits")


def get_within_event_split(chip_table, test_frac=0.2, val_frac=0.2, seed=42):
    """Chip-level reference split over all 804 chips (unit = chip; LEGACY).

    M1.6 note: this is NOT a "within-event" protocol. Every event has
    exactly one chip in hls_burn_scars, so a chip-level random split is
    automatically event-disjoint (no event contributes to two splits).
    Retained as a reference manifest only; it is not the primary
    comparison population (matched-Wildfire events only, see
    get_kfold_event_split) and must not be used for the primary experiment.

    chip_table: DataFrame with at least an 'image_id' column.
    Returns DataFrame with columns [image_id, split].
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(chip_table["image_id"].unique()))
    rng.shuffle(ids)
    n_test = int(round(test_frac * len(ids)))
    n_val = int(round(val_frac * len(ids)))
    roles = (["test"] * n_test + ["val"] * n_val
             + ["train"] * (len(ids) - n_test - n_val))
    return pd.DataFrame({"image_id": ids, "split": roles})


def get_leave_one_event_out_split(chip_table, held_out_event):
    """LOEO for one held-out event.

    chip_table: DataFrame with 'image_id', 'event_id' (event_id must be
    non-null for matched rows). Rows without a usable event_id are excluded
    from LOEO entirely (they cannot be assigned to any fold).
    Returns (train_val, test) DataFrames with columns [image_id, event_id].
    """
    usable = chip_table[chip_table["event_id"].notna()].copy()
    test = usable[usable["event_id"] == held_out_event]
    train_val = usable[usable["event_id"] != held_out_event]
    return train_val, test


def loeo_manifest(chip_table, seed=42):
    """Deterministic LOEO manifest for every usable event.

    One long DataFrame: image_id, event_id, held_out_event, role
    (train | test). The same seed shuffles the train pool per fold for
    reproducibility of downstream train/val splitting.

    M1.6 note: retained as an artifact; superseded as the PRIMARY protocol
    by 5-fold event-disjoint CV (get_kfold_event_split). LOEO remains a
    valid audit structure (one test event per fold).
    """
    usable = chip_table[chip_table["event_id"].notna()].copy()
    events = sorted(usable["event_id"].unique())
    rows = []
    for ev_id in events:
        tr, te = get_leave_one_event_out_split(chip_table, ev_id)
        for _, r in tr.iterrows():
            rows.append({"image_id": r["image_id"], "event_id": r["event_id"],
                         "held_out_event": ev_id, "role": "train"})
        for _, r in te.iterrows():
            rows.append({"image_id": r["image_id"], "event_id": r["event_id"],
                         "held_out_event": ev_id, "role": "test"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M1.6: primary protocol — event-disjoint K-fold CV (ONE shared assignment)
# ---------------------------------------------------------------------------

def build_overlap_components(pairs, nodes):
    """Union-find connected components of an undirected graph.

    pairs: iterable of (a, b) pairs (overlapping event pairs).
    nodes: iterable of all nodes.
    Returns components (list of lists of nodes), sorted by size desc then
    by the sorted node list (deterministic).
    """
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    return sorted((sorted(c) for c in comps.values()),
                  key=lambda c: (-len(c), c))


def _long_kfold_manifest(event_rows, fold_of, k):
    """Expand a fold assignment into the long manifest.

    event_rows: one row per event with [event_id, image_id].
    fold_of: dict event_id -> fold (1..k).
    Returns DataFrame [image_id, event_id, fold, role]: for fold f,
    test = events of fold f, train = all other events. Each event appears
    exactly k times (1 test row + k-1 train rows).
    """
    rows = []
    for _, r in event_rows.iterrows():
        eid = r["event_id"]
        for f in range(1, k + 1):
            rows.append({
                "image_id": r["image_id"],
                "event_id": eid,
                "fold": f,
                "role": "test" if fold_of[eid] == f else "train",
            })
    return pd.DataFrame(rows)


def _greedy_assign(groups, k):
    """Assign groups (largest-first) to the currently smallest fold.

    groups: list of lists (each group's members go to the same fold).
    Returns (fold_of: dict member -> fold 1..k, fold_sizes: list).
    Ties in size go to the lowest fold index. Deterministic.
    """
    groups = sorted(groups, key=lambda g: (-len(g), [str(m) for m in g]))
    fold_sizes = [0] * k
    fold_of = {}
    for group in groups:
        f = int(np.argmin(fold_sizes))
        for member in group:
            fold_of[member] = f + 1
        fold_sizes[f] += len(group)
    return fold_of, fold_sizes


def get_kfold_event_split(event_rows, k=5, seed=42):
    """PRIMARY protocol: deterministic event-disjoint k-fold CV manifest.

    event_rows: DataFrame, one row per event, with columns
    [event_id, image_id, fire_year, burned_fraction] (event == chip in
    hls_burn_scars, so the event's single image_id is the fold unit).

    Recorded procedure:
    1. burned_fraction_bin = 4 quantile bins of burned_fraction (qcut).
    2. strata = (fire_year, burned_fraction_bin); processed largest-first.
    3. Within each stratum, events are shuffled with rng(seed) and assigned
       one-by-one to the currently smallest fold (ties -> lowest index).
    This keeps fold sizes nearly equal while stratifying on fire year and
    burned fraction — no fragile heuristics. The result is ONE shared
    assignment; both models must use exactly these folds (no
    model-specific folds).

    Returns (manifest [image_id, event_id, fold, role], bin_edges list).
    """
    rows = event_rows.copy()
    codes, edges = pd.qcut(rows["burned_fraction"], 4, labels=False,
                           retbins=True, duplicates="drop")
    rows["burned_fraction_bin"] = codes
    strata = (rows.groupby(["fire_year", "burned_fraction_bin"], sort=False)
              ["event_id"].agg(list))
    strata = strata.sort_values(key=lambda s: s.map(len), ascending=False)
    rng = np.random.default_rng(seed)
    fold_sizes = [0] * k
    fold_of = {}
    for ids_ in strata:
        ids_s = list(ids_)
        rng.shuffle(ids_s)
        for eid in ids_s:
            f = int(np.argmin(fold_sizes))
            fold_of[eid] = f + 1
            fold_sizes[f] += 1
    manifest = _long_kfold_manifest(rows, fold_of, k)
    return manifest, list(edges)


def get_spatial_grouped_kfold(event_rows, components, k=5):
    """Sensitivity: spatial-overlap-grouped k-fold CV.

    components: connected components of the event-footprint overlap graph
    (list of lists of event_ids). All events of one component MUST share a
    fold — so no train/test pair has spatially intersecting footprints.
    Components are assigned largest-first to the currently smallest fold.

    event_rows: one row per event with [event_id, image_id].
    Returns the long-format manifest [image_id, event_id, fold, role].
    """
    nodes = set(event_rows["event_id"])
    comps = [c for c in components if set(c).issubset(nodes)]
    assert len(set().union(*comps)) == len(nodes), "components miss events"
    fold_of, _ = _greedy_assign(comps, k)
    return _long_kfold_manifest(event_rows, fold_of, k)


def get_tile_grouped_kfold(event_rows, k=5):
    """Sensitivity (stricter, NOT primary): HLS-tile-disjoint k-fold CV.

    Events sharing an HLS tile form a group; groups are assigned
    largest-first to the currently smallest fold. Stricter than footprint
    overlap: two events in the same 110 km tile share a fold even when
    their 15 km chips do not intersect.

    event_rows: one row per event with [event_id, image_id, hls_tile].
    Returns the long-format manifest [image_id, event_id, fold, role].
    """
    tiles = {}
    for _, r in event_rows.iterrows():
        tiles.setdefault(r["hls_tile"], []).append(r["event_id"])
    fold_of, _ = _greedy_assign(list(tiles.values()), k)
    return _long_kfold_manifest(event_rows, fold_of, k)


def assert_no_cross_fold_overlap(manifest, overlap_pairs):
    """Assert zero train/test footprint overlap in every fold.

    overlap_pairs: DataFrame with columns [event_a, event_b]; each row is a
    pair of events whose footprints intersect. Returns the list of
    violations [(event_a, event_b, fold_a, fold_b)] — empty iff every
    overlapping pair shares a fold (and therefore never crosses the
    train/test boundary). Callers raise on non-empty.
    """
    tests = manifest[manifest["role"] == "test"]
    fold_of = (tests.drop_duplicates("event_id")
               .set_index("event_id")["fold"].to_dict())
    violations = []
    for _, r in overlap_pairs.iterrows():
        a, b = r["event_a"], r["event_b"]
        if a in fold_of and b in fold_of and fold_of[a] != fold_of[b]:
            violations.append((a, b, fold_of[a], fold_of[b]))
    return violations


def get_inner_split(primary_rows, outer_fold_of, val_frac=0.2, seed=42,
                    n_bins=4):
    """M2 frozen inner protocol: event-disjoint inner train/val WITHIN each
    outer training pool.

    The outer 5-fold assignment (``outer_fold_of``: event_id -> outer fold 1..k)
    is fixed by M1.6 and shared by both models. For every outer fold f this
    builds the inner split of that fold's outer-train pool:

      * split unit = EVENT (= chip here); no pixel-level split;
      * an event appears in at most one of {inner-train, inner-val} per outer
        fold, and never in the outer test set of that fold;
      * outer test events never influence the inner split (they are excluded
        from the pool before stratification);
      * stratification = (fire_year x burned-fraction quantile bin) using the
        SAME qcut binning as the outer split (recomputed on the same primary
        rows, so bin edges are identical to M1.6);
      * within each stratum, events are shuffled with rng([seed, f]) and the
        first round(val_frac * n) events go to inner-val, the rest to
        inner-train. No fragile heuristics — small strata simply contribute
        zero or one val event.

    Returns a long manifest [image_id, event_id, outer_fold, role] with
    role in {"test", "train", "val"}: "test" = the outer fold's held-out
    events, "train"/"val" = inner split of the outer-train pool. Each event
    appears exactly k times (1 test + k-1 train/val). Deterministic given
    (primary_rows, outer_fold_of, seed).
    """
    rows = primary_rows.copy()
    codes, _edges = pd.qcut(rows["burned_fraction"], n_bins, labels=False,
                            retbins=True, duplicates="drop")
    rows["burned_fraction_bin"] = codes

    out_rows = []
    for f in sorted(set(outer_fold_of.values())):
        test_ids = {eid for eid, ff in outer_fold_of.items() if ff == f}
        pool = rows[~rows["event_id"].isin(test_ids)]
        strata = (pool.groupby(["fire_year", "burned_fraction_bin"], sort=True)
                  ["event_id"].apply(list))
        rng = np.random.default_rng([seed, f])
        inner_role = {}
        for ids_ in strata:
            ids_s = list(ids_)
            rng.shuffle(ids_s)
            n_val = int(round(val_frac * len(ids_s)))
            for i, eid in enumerate(ids_s):
                inner_role[eid] = "val" if i < n_val else "train"
        for _, r in rows.iterrows():
            eid = r["event_id"]
            out_rows.append({
                "image_id": r["image_id"],
                "event_id": eid,
                "outer_fold": f,
                "role": "test" if eid in test_ids else inner_role[eid],
            })
    return pd.DataFrame(out_rows)


def write_manifests(chip_table, seed=42):
    """Write the LEGACY chip-level + LOEO manifests (kept for reference).

    Primary M1.6 manifests are written by scripts/m1_6_make_protocol.py.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    within = get_within_event_split(chip_table, seed=seed)
    within.to_csv(os.path.join(SPLITS_DIR, f"split_within_seed{seed}.csv"),
                  index=False)
    loeo = loeo_manifest(chip_table, seed=seed)
    loeo.to_csv(os.path.join(SPLITS_DIR, f"split_loeo_seed{seed}.csv"),
                index=False)
    with open(os.path.join(SPLITS_DIR, "split_manifest_info.json"), "w",
              encoding="utf-8") as f:
        import json
        json.dump({
            "seed": seed,
            "conventional": {
                "unit": "chip", "test_frac": 0.2, "val_frac": 0.2,
                "n_chips": len(within),
                "file": f"split_within_seed{seed}.csv",
            },
            "loeo": {
                "unit": "event", "n_events": int(chip_table[
                    chip_table["event_id"].notna()]["event_id"].nunique()),
                "file": f"split_loeo_seed{seed}.csv",
                "note": "rows without event_id are excluded from LOEO",
            },
        }, f, indent=2)
    return within, loeo

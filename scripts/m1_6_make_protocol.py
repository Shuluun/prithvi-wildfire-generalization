"""M1.6: corrected experimental protocol (PROTOCOL CORRECTION ONLY).

Builds (no model training, no feature extraction, no Prithvi weights):
- primary analysis population (matched events with MTBS Incid_Type ==
  'Wildfire') -> data/metadata/event_attributes.csv
- legacy-manifest audit -> results/reports/m1_6_legacy_manifest_audit.csv
- primary-population overlap graph + spatial components
  -> results/reports/event_overlap_pairs_primary.csv,
     results/reports/spatial_components_primary.csv
- three shared 5-fold event-disjoint manifests (seed 42):
  * split_k5_event_seed42.csv      (PRIMARY, stratified)
  * split_k5_spatial_seed42.csv    (spatial-clean sensitivity;
                                    zero cross-fold footprint overlap)
  * split_k5_tiledisjoint_seed42.csv (stricter sensitivity, NOT primary)
  -> data/metadata/splits/ (+ updated split_manifest_info.json)
- protocol stats -> results/reports/m1_6_protocol_stats.json

Usage (from repo root):
  python scripts/m1_6_make_protocol.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.strtree import STRtree
from shapely.ops import transform as shapely_transform
import pyproj
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT, MTBS_SHP_DIR, RESULTS_ROOT  # noqa: E402
from src.data.splits import (  # noqa: E402
    assert_no_cross_fold_overlap, build_overlap_components,
    get_kfold_event_split, get_spatial_grouped_kfold, get_tile_grouped_kfold,
)

SEED = 42
K = 5
SPLITS_DIR = os.path.join(METADATA_ROOT, "splits")

CRS_WGS84 = pyproj.CRS.from_epsg(4326)
CRS_5070 = pyproj.CRS.from_epsg(5070)


def load_mtbs_incid_type():
    """Incid_Type per MTBS Event_ID (authoritative event-type source)."""
    gdf = gpd.read_file(os.path.join(MTBS_SHP_DIR, "mtbs_perims_DD.shp"))
    out = gdf[["Event_ID", "Incid_Type"]].copy()
    out["event_id"] = out["Event_ID"].astype(str)
    out = out[["event_id", "Incid_Type"]].rename(
        columns={"Incid_Type": "incid_type"})
    return out


def build_event_attributes(events, inventory, incid):
    """One row per matched event: identity, type, chip stats, burn fraction."""
    matched = events[events["match_status"] == "matched"].copy()
    att = matched.merge(inventory[["image_id", "split", "burned_pixels",
                                   "unburned_pixels", "missing_pixels"]],
                        on="image_id", how="left")
    att = att.merge(incid, on="event_id", how="left")
    att["burned_fraction"] = (
        att["burned_pixels"] / (att["burned_pixels"] + att["unburned_pixels"]))
    cols = ["event_id", "image_id", "hls_tile", "fire_name", "fire_year",
            "incid_type", "split", "burned_pixels", "unburned_pixels",
            "missing_pixels", "burned_fraction", "acquisition_date",
            "chip_bounds"]
    return att[cols]


def compute_overlap_pairs(primary):
    """Recompute pairwise footprint intersections (WGS84 -> EPSG:5070),
    independent of the M1.5 audit CSV."""
    project = Transformer.from_crs(CRS_WGS84, CRS_5070,
                                   always_xy=True).transform
    foot = {}
    for _, r in primary.iterrows():
        geom = shapely_transform(project, wkt.loads(r["chip_bounds"]))
        foot[r["event_id"]] = geom
    ids = list(foot.keys())
    tree = STRtree([foot[i] for i in ids])
    pairs = []
    for i, eid in enumerate(ids):
        for j in tree.query(foot[eid]):
            if j <= i:
                continue
            other = ids[j]
            inter = foot[eid].intersection(foot[other]).area
            if inter > 0:
                pairs.append({"event_a": eid, "event_b": other,
                              "overlap_area_m2": round(inter, 1)})
    return pd.DataFrame(pairs), foot


def fold_sizes_of(manifest):
    return (manifest[manifest["role"] == "test"].groupby("fold")
            ["event_id"].nunique().sort_index().tolist())


def audit_legacy_manifests(ev):
    """Task 1: match-status composition of the LEGACY manifests."""
    rows = []
    within = pd.read_csv(os.path.join(SPLITS_DIR, "split_within_seed42.csv"))
    wm = within.merge(ev[["image_id", "match_status", "event_id"]],
                      on="image_id", how="left")
    for role in ("train", "val", "test"):
        sub = wm[wm["split"] == role]
        row = {"manifest": "split_within_seed42", "role": role,
               "n_chips": len(sub)}
        for st in ("matched", "ambiguous", "unmatched"):
            row[f"n_{st}"] = int((sub["match_status"] == st).sum())
        rows.append(row)
    rows.append({
        "manifest": "split_within_seed42", "role": "(all)", "n_chips": len(wm),
        "events_in_multiple_roles": int(
            (wm.groupby("event_id")["split"].nunique() > 1).sum()),
        "note": "0 => chip-level split is automatically event-disjoint "
                "(1 chip per event); it is NOT a within-event protocol",
    })
    loeo = pd.read_csv(os.path.join(SPLITS_DIR, "split_loeo_seed42.csv"))
    rows.append({
        "manifest": "split_loeo_seed42", "role": "(all)",
        "n_chips": int(loeo["image_id"].nunique()),
        "n_events": int(loeo["event_id"].nunique()),
        "n_folds": int(loeo["held_out_event"].nunique()),
        "note": "LOEO artifact; superseded as PRIMARY protocol by K5",
    })
    return pd.DataFrame(rows)


def main():
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    inv = pd.read_csv(os.path.join(METADATA_ROOT, "chip_inventory.csv"))
    incid = load_mtbs_incid_type()
    att = build_event_attributes(ev, inv, incid)
    os.makedirs(METADATA_ROOT, exist_ok=True)
    att.to_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"),
               index=False)
    print("event_attributes (matched) by Incid_Type:")
    print(att["incid_type"].value_counts(dropna=False).to_string())
    n_missing = int(att["incid_type"].isna().sum())
    assert n_missing == 0, f"{n_missing} matched events missing Incid_Type"

    primary = att[att["incid_type"] == "Wildfire"].copy()
    n_primary = len(primary)
    assert primary["event_id"].is_unique and primary["image_id"].is_unique
    print(f"\nPRIMARY ANALYSIS POPULATION (Wildfire): {n_primary} events "
          f"(1 chip each)")

    # --- task 1: legacy manifest audit ---
    audit = audit_legacy_manifests(ev)
    os.makedirs(os.path.join(RESULTS_ROOT, "reports"), exist_ok=True)
    audit.to_csv(os.path.join(RESULTS_ROOT, "reports",
                              "m1_6_legacy_manifest_audit.csv"), index=False)
    print("\nlegacy manifest audit:")
    print(audit.to_string(index=False))

    # --- task 5: overlap graph over the primary population ---
    pairs_df, _ = compute_overlap_pairs(primary)
    print(f"\nprimary overlap pairs (recomputed from footprints): "
          f"{len(pairs_df)}")
    # cross-check against the M1.5 audit CSV restricted to primary events
    old = pd.read_csv(os.path.join(RESULTS_ROOT, "reports",
                                   "event_overlap_pairs.csv"))
    pid = set(primary["event_id"])
    old_primary = old[old["event_a"].isin(pid) & old["event_b"].isin(pid)]
    set_old = {frozenset((r.event_a, r.event_b)) for r in old_primary.itertuples()}
    set_new = {frozenset((r.event_a, r.event_b)) for r in pairs_df.itertuples()}
    assert set_old == set_new, "recomputed pairs disagree with the M1.5 audit"
    print(f"cross-check OK: matches the M1.5 audit restricted to primary "
          f"({len(old_primary)} pairs)")

    pairs_df.to_csv(os.path.join(RESULTS_ROOT, "reports",
                                 "event_overlap_pairs_primary.csv"),
                    index=False)
    comps = build_overlap_components(
        [(r.event_a, r.event_b) for r in pairs_df.itertuples()],
        list(pid))
    comp_df = pd.DataFrame(
        [{"component_id": i + 1, "n_events": len(c),
          "event_ids": ";".join(sorted(c))} for i, c in enumerate(comps)])
    comp_df.to_csv(os.path.join(RESULTS_ROOT, "reports",
                                "spatial_components_primary.csv"), index=False)
    comp_sizes = [len(c) for c in comps]
    print(f"components: {len(comps)} | singletons: "
          f"{sum(1 for s in comp_sizes if s == 1)} | max size: "
          f"{max(comp_sizes)}")

    # --- tasks 4/5: the three shared K5 manifests ---
    k5_event, bin_edges = get_kfold_event_split(primary, k=K, seed=SEED)
    k5_event2, _ = get_kfold_event_split(primary, k=K, seed=SEED)
    pd.testing.assert_frame_equal(k5_event, k5_event2)  # deterministic
    k5_spatial = get_spatial_grouped_kfold(primary, comps, k=K)
    k5_tile = get_tile_grouped_kfold(primary, k=K)

    # zero-overlap proof (both pair sets: recomputed + M1.5 audit)
    vio = assert_no_cross_fold_overlap(k5_spatial, pairs_df)
    vio_old = assert_no_cross_fold_overlap(k5_spatial, old_primary)
    assert not vio and not vio_old, f"spatial manifest leaks: {vio[:5]}"
    print("\nspatial manifest: ZERO cross-fold footprint overlap "
          f"(checked {len(pairs_df)} pairs, recomputed + audit)")

    vio_event = assert_no_cross_fold_overlap(k5_event, pairs_df)
    print(f"event K5 (random): {len(vio_event)} overlapping pairs cross the "
          f"train/test boundary — the contrast the sensitivity quantifies")

    sizes = {
        "k5_event": fold_sizes_of(k5_event),
        "k5_spatial": fold_sizes_of(k5_spatial),
        "k5_tile": fold_sizes_of(k5_tile),
    }
    for name, s in sizes.items():
        print(f"{name} fold sizes: {s} (ratio "
              f"{max(s) / min(s):.3f})")

    k5_event.to_csv(os.path.join(SPLITS_DIR, "split_k5_event_seed42.csv"),
                    index=False)
    k5_spatial.to_csv(os.path.join(SPLITS_DIR, "split_k5_spatial_seed42.csv"),
                      index=False)
    k5_tile.to_csv(os.path.join(SPLITS_DIR,
                                "split_k5_tiledisjoint_seed42.csv"),
                   index=False)

    n_tiles = primary["hls_tile"].nunique()
    tile_group_sizes = primary.groupby("hls_tile")["event_id"].nunique()

    info = {
        "updated": "2026-08-31 (M1.6 protocol correction)",
        "seed": SEED,
        "k": K,
        "split_hierarchy": "EVENT -> CHIP -> PIXEL; event == chip in "
                           "hls_burn_scars (1 chip per matched event)",
        "primary_population": {
            "definition": "matched events with MTBS Incid_Type == 'Wildfire'",
            "n_events": n_primary,
            "source": "data/metadata/event_attributes.csv",
            "excluded_from_primary": "Prescribed Fire (177, kept for "
                                     "secondary/domain-shift), Unknown (3), "
                                     "ambiguous (24) + unmatched (24) chips",
        },
        "legacy_manifests": {
            "split_within_seed42.csv": "chip-level reference over all 804 "
                "chips; NOT within-event (0 events in >1 role); NOT primary",
            "split_loeo_seed42.csv": "756-fold LOEO artifact; superseded as "
                "primary protocol by the K5 manifests",
        },
        "k5_event": {
            "file": "split_k5_event_seed42.csv", "role": "PRIMARY",
            "shared_by": ["spectral RF", "frozen Prithvi"],
            "procedure": "strata = (fire_year, 4 quantile bins of "
                "burned_fraction); strata processed largest-first; within a "
                "stratum events are rng(42)-shuffled and assigned to the "
                "currently smallest fold (ties -> lowest index)",
            "burned_fraction_bin_edges": [round(e, 6) for e in bin_edges],
            "fold_sizes": sizes["k5_event"],
        },
        "k5_spatial": {
            "file": "split_k5_spatial_seed42.csv",
            "role": "spatial-generalization sensitivity",
            "procedure": "connected components of the primary event-"
                "footprint overlap graph; components assigned largest-first "
                "to the currently smallest fold; overlapping events always "
                "share a fold",
            "n_components": len(comps),
            "n_singletons": int(sum(1 for s in comp_sizes if s == 1)),
            "max_component_size": int(max(comp_sizes)),
            "n_overlap_pairs_checked": len(pairs_df),
            "zero_cross_fold_overlap": True,
            "fold_sizes": sizes["k5_spatial"],
        },
        "k5_tiledisjoint": {
            "file": "split_k5_tiledisjoint_seed42.csv",
            "role": "stricter sensitivity (HLS-tile-disjoint); NOT primary",
            "procedure": "events sharing an HLS tile form a group; groups "
                "assigned largest-first to the currently smallest fold",
            "n_tiles": int(n_tiles),
            "max_tile_group_size": int(tile_group_sizes.max()),
            "fold_sizes": sizes["k5_tile"],
        },
    }
    with open(os.path.join(SPLITS_DIR, "split_manifest_info.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    stats = dict(info)
    stats["legacy_manifest_audit"] = audit.to_dict(orient="records")
    stats["matched_incid_type_counts"] = (
        att["incid_type"].value_counts(dropna=False).to_dict())
    with open(os.path.join(RESULTS_ROOT, "reports",
                           "m1_6_protocol_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nmanifests + stats written ({SPLITS_DIR}, "
          f"{RESULTS_ROOT}/reports)")


if __name__ == "__main__":
    main()

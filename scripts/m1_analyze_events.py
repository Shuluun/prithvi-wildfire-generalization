"""M1: per-event statistics + geographic-overlap audit from classified events.

Inputs: data/metadata/events.csv (classified) + data/metadata/chip_inventory.csv
Outputs (results/reports/):
  events_summary.csv  — per-event: chips, burned/unburned/missing pixels,
                        acquisition-date range, footprint (union of chip boxes)
  event_overlap_pairs.csv — intersecting event pairs with overlap area
  stdout: gate numbers (matched/ambiguous/unmatched, unique events, events
          with >=2 chips, etc.)
"""
import os
import sys

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.strtree import STRtree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT, RESULTS_ROOT  # noqa: E402


def main():
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    inv = pd.read_csv(os.path.join(METADATA_ROOT, "chip_inventory.csv"))
    merged = ev.merge(inv[["image_id", "split", "burned_pixels",
                           "unburned_pixels", "missing_pixels"]],
                      on="image_id", how="left")

    matched = merged[merged["match_status"] == "matched"]
    out = []
    for event_id, grp in matched.groupby("event_id"):
        out.append({
            "event_id": event_id,
            "fire_name": grp["fire_name"].iloc[0],
            "fire_year": grp["fire_year"].iloc[0],
            "n_chips": len(grp),
            "n_train_chips": int((grp["split"] == "training").sum()),
            "n_val_chips": int((grp["split"] == "validation").sum()),
            "burned_pixels": int(grp["burned_pixels"].sum()),
            "unburned_pixels": int(grp["unburned_pixels"].sum()),
            "missing_pixels": int(grp["missing_pixels"].sum()),
            "acq_min": grp["acquisition_date"].min(),
            "acq_max": grp["acquisition_date"].max(),
            "temporal_gap_days_mean": round(grp["temporal_gap_days"].mean(), 1),
            "match_confidence_min": round(grp["match_confidence"].min(), 4),
            "burn_mask_overlap_min": round(grp["burn_mask_overlap"].min(), 4),
        })
    summ = pd.DataFrame(out).sort_values("n_chips", ascending=False)
    os.makedirs(os.path.join(RESULTS_ROOT, "reports"), exist_ok=True)
    summ.to_csv(os.path.join(RESULTS_ROOT, "reports", "events_summary.csv"),
                index=False)

    # Geographic overlap between events: chip_bounds in events.csv are WGS84
    # (EPSG:4326); reproject to EPSG:5070 (CONUS Albers) for metric areas.
    from pyproj import Transformer
    import pyproj
    from shapely.ops import unary_union, transform as shapely_transform

    print("\nreprojecting chip footprints (WGS84 -> EPSG:5070) for overlap audit ...")
    project = Transformer.from_crs(pyproj.CRS.from_epsg(4326),
                                   pyproj.CRS.from_epsg(5070),
                                   always_xy=True).transform
    # Overlap audit is event-level: matched rows only (ambiguous/unmatched rows
    # have event_id = NaN and must not be pooled into a fake "event").
    footprints = {}
    for _, r in matched.iterrows():
        geom = shapely_transform(project, wkt.loads(r["chip_bounds"]))
        footprints.setdefault(r["event_id"], []).append(geom)

    event_geoms = {eid: unary_union(gs) for eid, gs in footprints.items()}
    ids = list(event_geoms.keys())
    tree = STRtree([event_geoms[i] for i in ids])
    pairs = []
    for i, eid in enumerate(ids):
        hits = tree.query(event_geoms[eid])
        for j in hits:
            if j <= i:
                continue
            other = ids[j]
            inter = event_geoms[eid].intersection(event_geoms[other]).area
            if inter > 0:
                pairs.append({
                    "event_a": eid, "event_b": other,
                    "overlap_area_m2": round(inter, 1),
                })
    ov = pd.DataFrame(pairs)
    ov.to_csv(os.path.join(RESULTS_ROOT, "reports",
                           "event_overlap_pairs.csv"), index=False)

    # Gate numbers
    n_events = len(summ)
    n_events_multi = int((summ["n_chips"] >= 2).sum())
    n_events_5plus = int((summ["n_chips"] >= 5).sum())
    print("\n=== M1 gate numbers ===")
    print(f"total chips: {len(ev)}")
    print(f"matched: {(merged['match_status']=='matched').sum()} | "
          f"ambiguous: {(merged['match_status']=='ambiguous').sum()} | "
          f"unmatched: {(merged['match_status']=='unmatched').sum()}")
    print(f"unique events (matched): {n_events}")
    print(f"events with >=2 chips: {n_events_multi}; with >=5 chips: "
          f"{n_events_5plus}")
    print(f"chips per event: min=1 median="
          f"{summ['n_chips'].median()} max={summ['n_chips'].max()}")
    print(f"events in both splits: "
          f"{int(((summ['n_train_chips']>0)&(summ['n_val_chips']>0)).sum())}")
    print(f"overlapping event pairs: {len(ov)}")
    print(f"\nambiguous rows:")
    amb = merged[merged["match_status"] == "ambiguous"]
    if len(amb):
        print(amb[["image_id", "fire_name", "match_confidence",
                   "intersection_ratio", "burn_mask_overlap"]]
              .head(30).to_string(index=False))
    print(f"\nunmatched rows:")
    unm = merged[merged["match_status"] == "unmatched"]
    if len(unm):
        print(unm[["image_id", "hls_tile", "acquisition_date"]].head(30)
              .to_string(index=False))


if __name__ == "__main__":
    main()

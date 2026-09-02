"""M1.6: ambiguous-case QC — top-2 candidate perimeters per chip (QC ONLY).

For every ambiguous chip (match_status == 'ambiguous' in events.csv):
- re-rank ALL spatial+temporal candidates (chip_candidates);
- draw the TOP-2 MTBS perimeters on the burn-mask panel (candidate 1:
  yellow solid, candidate 2: magenta dashed) + RGB panel + text panel with
  candidate IDs, score, burn_mask_overlap, intersection_ratio, temporal
  gap and score margin;
- save figures to results/figures/qc/ambiguous_top2/ and the numbers to
  results/reports/ambiguous_top2_candidates.csv.

This is diagnostics ONLY: it does not alter any matching threshold and
does not change events.csv.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from m1_5_visual_qc import load_rgb  # noqa: E402
from src.data.event_reconstruction import (  # noqa: E402
    _reproject_geom, chip_candidates, chip_crs_of, load_perimeters,
)
from src.data.paths import (  # noqa: E402
    HLS_BURN_SCARS_DIR, METADATA_ROOT, MTBS_SHP_DIR, RESULTS_ROOT,
)

OUT_DIR = os.path.join(RESULTS_ROOT, "figures", "qc", "ambiguous_top2")

C1_COLOR = (1.0, 1.0, 0.0)     # yellow: candidate 1 (solid)
C2_COLOR = (1.0, 0.0, 1.0)     # magenta: candidate 2 (dashed)


def draw_perimeter(ax, gdf, event_id, chip_crs, inv, color, dashed):
    row = gdf[gdf["Event_ID"] == event_id]
    if row.empty:
        return 0
    poly = _reproject_geom(row.geometry.iloc[0], gdf.crs, chip_crs)
    geoms = (list(poly.geoms) if poly.geom_type == "MultiPolygon"
             else [poly])
    n = 0
    for g in geoms:
        xy = np.asarray([inv * (x, y) for x, y in g.exterior.coords])
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, color=color,
                                linewidth=1.3,
                                linestyle="--" if dashed else "-"))
        n += 1
    return n


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    amb = ev[ev["match_status"] == "ambiguous"].copy()
    print(f"ambiguous chips: {len(amb)}")

    perimeters_gdf, perimeters_tree = load_perimeters(MTBS_SHP_DIR)
    rows_out = []
    verified = []
    for _, chip_row in amb.iterrows():
        image_id = chip_row["image_id"]
        split = None
        for sdir in ("training", "validation"):
            p = os.path.join(HLS_BURN_SCARS_DIR, sdir, f"{image_id}_merged.tif")
            if os.path.exists(p):
                split = sdir
                break
        merged_p = os.path.join(HLS_BURN_SCARS_DIR, split,
                                f"{image_id}_merged.tif")
        mask_p = merged_p.replace("_merged.tif", ".mask.tif")

        _iid, _tile, _date, _fp, ranked = chip_candidates(
            merged_p, mask_p, perimeters_gdf, perimeters_tree)
        top2 = ranked[:2]

        rgb = load_rgb(merged_p)
        with rasterio.open(mask_p) as mds:
            mask = mds.read(1)
            crs = chip_crs_of(mds)
            inv = ~mds.transform

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB (B04,B03,B02)", fontsize=8)
        axes[0].axis("off")
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        mask_rgb[mask == 1] = (1.0, 0.2, 0.2)
        mask_rgb[mask == 0] = (0.35, 0.35, 0.35)
        mask_rgb[mask == -1] = (0, 0, 0)
        axes[1].imshow(mask_rgb)
        n1 = draw_perimeter(axes[1], perimeters_gdf, top2[0]["event_id"],
                            crs, inv, C1_COLOR, dashed=False)
        n2 = 0
        if len(top2) > 1:
            n2 = draw_perimeter(axes[1], perimeters_gdf, top2[1]["event_id"],
                                crs, inv, C2_COLOR, dashed=True)
        axes[1].set_title("mask + top-2 candidates", fontsize=8)
        axes[1].axis("off")

        # text panel: candidate diagnostics
        lines = [f"chip: {image_id}", "",
                 f"C1 {top2[0]['event_id']}",
                 f"  {top2[0]['fire_name'][:26]} "
                 f"({top2[0]['incid_type']})",
                 f"  ig {fmt(top2[0]['ig_date'])}",
                 f"  score {fmt(top2[0]['score'])} | "
                 f"ovl {fmt(top2[0]['burn_overlap'])}",
                 f"  inter {fmt(top2[0]['inter_ratio'])} | "
                 f"gap {fmt(top2[0]['gap'])} d"]
        margin = None
        if len(top2) > 1:
            margin = top2[0]["score"] - top2[1]["score"]
            lines += [f"  margin {margin:.3f}", "",
                      f"C2 {top2[1]['event_id']}",
                      f"  {top2[1]['fire_name'][:26]} "
                      f"({top2[1]['incid_type']})",
                      f"  ig {fmt(top2[1]['ig_date'])}",
                      f"  score {fmt(top2[1]['score'])} | "
                      f"ovl {fmt(top2[1]['burn_overlap'])}",
                      f"  inter {fmt(top2[1]['inter_ratio'])} | "
                      f"gap {fmt(top2[1]['gap'])} d"]
        lines += ["", f"n_candidates {len(ranked)}",
                  f"chip conf {chip_row['match_confidence']:.3f} [ambiguous]"]
        axes[2].axis("off")
        axes[2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                     fontsize=7, family="monospace")
        fig.suptitle(f"AMBIGUOUS QC — {image_id[:70]}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"amb_{image_id}.png"), dpi=80)

        # programmatic verification: outline colors present in the rendered
        # canvas (human review unavailable in this run)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255
        yellow = int(((arr[:, :, 0] > 0.75) & (arr[:, :, 1] > 0.75)
                      & (arr[:, :, 2] < 0.45)).sum())
        magenta = int(((arr[:, :, 0] > 0.75) & (arr[:, :, 1] < 0.45)
                       & (arr[:, :, 2] > 0.75)).sum())
        verified.append((image_id, n1, n2, yellow, magenta))
        plt.close(fig)

        for rank, c in enumerate(top2, start=1):
            rows_out.append({
                "image_id": image_id,
                "chip_match_confidence": chip_row["match_confidence"],
                "n_candidates": len(ranked),
                "rank": rank,
                "candidate_event_id": c["event_id"],
                "fire_name": c["fire_name"],
                "incid_type": c["incid_type"],
                "ig_date": (c["ig_date"].isoformat()
                            if c["ig_date"] is not None else ""),
                "score": round(c["score"], 6),
                "burn_mask_overlap": (None if c["burn_overlap"] is None
                                      or np.isnan(c["burn_overlap"])
                                      else round(float(c["burn_overlap"]), 6)),
                "intersection_ratio": round(c["inter_ratio"], 6),
                "temporal_gap_days": c["gap"],
                "score_margin": (round(margin, 6) if rank == 1 and margin
                                 is not None else None),
            })

    csv_df = pd.DataFrame(rows_out)
    os.makedirs(os.path.join(RESULTS_ROOT, "reports"), exist_ok=True)
    csv_df.to_csv(os.path.join(RESULTS_ROOT, "reports",
                               "ambiguous_top2_candidates.csv"), index=False)

    n_bad = 0
    for image_id, n1, n2, yellow, magenta in verified:
        ok = yellow > 0 and (magenta > 0) == (n2 > 0)
        if not ok:
            n_bad += 1
            print(f"VERIFY FAIL {image_id}: n1={n1} n2={n2} "
                  f"yellow={yellow} magenta={magenta}")
    print(f"{len(verified)} figures -> {OUT_DIR}")
    print(f"verification: yellow outline present in all; magenta present "
          f"iff a 2nd candidate drawn — failures: {n_bad}")
    print(f"candidates CSV -> results/reports/ambiguous_top2_candidates.csv "
          f"({len(csv_df)} rows)")
    assert n_bad == 0


if __name__ == "__main__":
    main()

"""M1.5: visual QC — RGB composite + burn mask + matched MTBS perimeter.

Selection (seed 7): 16 matched (4 per year 2018-2021), 4 ambiguous,
4 unmatched = 24 figures in results/figures/qc/.
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
from src.data.event_reconstruction import _reproject_geom, chip_crs_of  # noqa: E402
from src.data.paths import (  # noqa: E402
    HLS_BURN_SCARS_DIR, METADATA_ROOT, MTBS_SHP_DIR, RESULTS_ROOT,
)

QC_DIR = os.path.join(RESULTS_ROOT, "figures", "qc")
RNG = np.random.default_rng(7)


def load_rgb(merged_path):
    with rasterio.open(merged_path) as ds:
        # band order in file: B02,B03,B04,B8A,B11,B12 (1-based)
        rgb = ds.read([3, 2, 1]).astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(rgb[i][rgb[i] > 0], (2, 98))
        band = np.clip(rgb[i], lo, hi)
        rgb[i] = (band - lo) / max(hi - lo, 1e-6)
    return np.transpose(rgb, (1, 2, 0))


def perimeter_outlines(mtbs_id, chip_crs, perimeters):
    """Return a list of exterior coordinate arrays (handles MultiPolygon)."""
    row = perimeters[perimeters["Event_ID"] == mtbs_id]
    if row.empty:
        return []
    poly = _reproject_geom(row.geometry.iloc[0], perimeters.crs, chip_crs)
    if poly.geom_type == "MultiPolygon":
        geoms = list(poly.geoms)
    else:
        geoms = [poly]
    return [np.asarray(g.exterior.coords) for g in geoms]


def main():
    os.makedirs(QC_DIR, exist_ok=True)
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    matched = ev[ev["match_status"] == "matched"]
    amb = ev[ev["match_status"] == "ambiguous"]
    unm = ev[ev["match_status"] == "unmatched"]
    pick = []
    for year in (2018, 2019, 2020, 2021):
        pool = matched[matched["fire_year"] == year]
        pick.extend(pool.sample(n=4, random_state=RNG.integers(1e6))
                    ["image_id"].tolist())
    pick.extend(amb["image_id"].tolist()[:4])
    pick.extend(unm["image_id"].tolist()[:4])

    perimeters = gpd.read_file(os.path.join(MTBS_SHP_DIR, "mtbs_perims_DD.shp"))

    for image_id in pick:
        row = ev[ev["image_id"] == image_id].iloc[0]
        # find which dir the chip is in (index files list both)
        split = None
        for sdir in ("training", "validation"):
            p = os.path.join(HLS_BURN_SCARS_DIR, sdir,
                             f"{image_id}_merged.tif")
            if os.path.exists(p):
                split = sdir
                break
        merged_p = os.path.join(HLS_BURN_SCARS_DIR, split,
                                f"{image_id}_merged.tif")
        mask_p = merged_p.replace("_merged.tif", ".mask.tif")
        rgb = load_rgb(merged_p)
        with rasterio.open(mask_p) as mds:
            mask = mds.read(1)
            crs = chip_crs_of(mds)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB (B04,B03,B02)", fontsize=8)
        axes[0].axis("off")
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
        mask_rgb[mask == 1] = (1.0, 0.2, 0.2)
        mask_rgb[mask == 0] = (0.35, 0.35, 0.35)
        mask_rgb[mask == -1] = (0, 0, 0)
        axes[1].imshow(mask_rgb)
        if row["match_status"] in ("matched", "ambiguous") and pd.notna(
                row["mtbs_id"]):
            # transform to pixel coords via inverse affine
            with rasterio.open(mask_p) as mds:
                inv = ~mds.transform
            for outline in perimeter_outlines(str(row["mtbs_id"]), crs,
                                              perimeters):
                xy = np.asarray([inv * (x, y) for x, y in outline])
                axes[1].add_patch(MplPolygon(
                    xy, closed=True, fill=False, edgecolor="yellow",
                    linewidth=1.0))
        status = row["match_status"]
        title = (f"{image_id[:60]}\n{row.get('fire_name', '')} "
                 f"({row.get('fire_year', '')}) [{status}]")
        axes[1].set_title("mask + perimeter", fontsize=8)
        axes[1].axis("off")
        fig.suptitle(title, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(QC_DIR, f"qc_{image_id}.png"), dpi=80)
        plt.close(fig)
    print(f"{len(pick)} QC figures -> {QC_DIR}")


if __name__ == "__main__":
    main()

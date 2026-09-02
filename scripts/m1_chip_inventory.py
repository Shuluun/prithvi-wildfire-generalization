"""M1: build the chip inventory (no MTBS needed).

For every chip in training/ and validation/: split, image_id, hls_tile,
acquisition_date (from granule ID), CRS (WKT), chip footprint (WKT), and
per-mask pixel counts (burned/unburned/missing).

Output: data/metadata/chip_inventory.csv — input for event reconstruction
QC, per-event statistics, and the M1.5 benchmark audit.
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.event_reconstruction import iter_chips, parse_granule  # noqa: E402
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT  # noqa: E402


def main():
    os.makedirs(METADATA_ROOT, exist_ok=True)
    rows = []
    for split_dir in ("training", "validation"):
        for merged, mask in iter_chips(HLS_BURN_SCARS_DIR, split_dir):
            image_id = os.path.basename(merged).replace("_merged.tif", "")
            tile, date = parse_granule(merged)
            with rasterio.open(merged) as ds:
                crs_wkt = ds.crs.to_wkt()
                b = ds.bounds
                bounds_wkt = f"POLYGON (({b.left} {b.bottom}, {b.left} {b.top}, " \
                             f"{b.right} {b.top}, {b.right} {b.bottom}, " \
                             f"{b.left} {b.bottom}))"
            with rasterio.open(mask) as mds:
                m = mds.read(1)
                vals, cnts = np.unique(m, return_counts=True)
                d = dict(zip(vals.tolist(), cnts.tolist()))
            rows.append({
                "split": split_dir,
                "image_id": image_id,
                "hls_tile": tile,
                "acquisition_date": date.isoformat(),
                "crs_wkt": crs_wkt,
                "chip_bounds": bounds_wkt,
                "burned_pixels": int(d.get(1, 0)),
                "unburned_pixels": int(d.get(0, 0)),
                "missing_pixels": int(d.get(-1, 0)),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(METADATA_ROOT, "chip_inventory.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} chips -> {out}")
    print(df.groupby("split").size().to_string())
    print(f"unique tiles: {df['hls_tile'].nunique()}, "
          f"date range: {df['acquisition_date'].min()} .. "
          f"{df['acquisition_date'].max()}")
    print("burned-pixel share per chip: "
          f"median={((df.burned_pixels / (df.burned_pixels + df.unburned_pixels)).median()):.3f}")


if __name__ == "__main__":
    main()

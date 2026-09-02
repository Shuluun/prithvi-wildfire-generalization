"""Event reconstruction — link hls_burn_scars chips to MTBS fire events.

Method:
- Read REAL georeferencing from every GeoTIFF: CRS, affine transform, bounds.
- Parse the acquisition date from the HLS granule ID (julian day) in the filename.
- Candidate matching: spatial intersection of the chip footprint with MTBS fire
  perimeters + temporal consistency (chip date >= ignition date) + burn-mask
  overlap (fraction of burned pixels inside the perimeter).
- Match confidence is a combination of intersection ratio and burn-mask overlap.
- Classification thresholds are NOT hard-coded guesses: they were finalized
  from the full 790-candidate distribution (the `pilot804` candidate stage over
  all 804 chips, `matching_summary_pilot804.txt`); the 200-chip pilot was the
  method-validation stage. See the M1 report §4 and the M1.6 reconciliation.

Output schema (14 fields):
  image_id, hls_tile, acquisition_date, chip_bounds, mtbs_id, fire_name,
  fire_year, intersection_area, intersection_ratio, burn_mask_overlap,
  temporal_gap_days, match_status, match_confidence, event_id
"""
import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import pyproj
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, MTBS_SHP_DIR  # noqa: E402

GRANULE_RE = re.compile(r"^subsetted_512x512_HLS\.S30\.T(\w{5})\.(\d{7})\.v(\d\.\d)(?:_merged)?\.tif$")

# Candidate filters (loose; only used to prune obviously irrelevant candidates).
MIN_INTERSECTION_RATIO = 0.01   # fraction of chip area inside a perimeter
MAX_TEMPORAL_GAP_DAYS = 3 * 365  # chip older than 3 years post-ignition is suspect

# Scoring weights (final thresholds chosen from the pilot distribution).
W_INTER = 0.5
W_BURN_OVERLAP = 0.5

COLUMNS = [
    "image_id", "hls_tile", "acquisition_date", "chip_bounds", "mtbs_id",
    "fire_name", "fire_year", "intersection_area", "intersection_ratio",
    "burn_mask_overlap", "temporal_gap_days", "match_status",
    "match_confidence", "second_score_margin", "post_id_prefix_match",
    "event_id",
]

# Canonical 14-field schema (events.csv); the other
# columns are pilot diagnostics kept only in the matching CSVs.
CANONICAL_COLUMNS = [c for c in COLUMNS
                     if c not in ("second_score_margin", "post_id_prefix_match")]


def parse_granule(filename):
    """Return (tile, date) from a chip filename, e.g.
    'subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged.tif' ->
    ('T10SDH', datetime.date(2020, 9, 4))."""
    base = os.path.basename(filename)
    m = GRANULE_RE.match(base)
    if not m:
        raise ValueError(f"filename not a recognized HLS granule: {base}")
    tile, doy, _ver = m.groups()
    date = dt.datetime.strptime(doy, "%Y%j").date()
    return tile, date


def chip_crs_of(ds):
    """Return a pyproj CRS for a rasterio dataset, even for custom WKT."""
    epsg = ds.crs.to_epsg()
    if epsg is not None:
        return pyproj.CRS.from_epsg(epsg)
    return pyproj.CRS.from_wkt(ds.crs.to_wkt())


def read_chip_geo(merged_path):
    """Read CRS + footprint (chip-CRS box) from a chip GeoTIFF."""
    with rasterio.open(merged_path) as ds:
        crs = chip_crs_of(ds)
        b = ds.bounds
        footprint = box(b.left, b.bottom, b.right, b.top)
    return crs, footprint


def _reproject_geom(geom, crs_from, crs_to):
    """Reproject any shapely geometry between CRSs."""
    from shapely.ops import transform as shapely_transform
    project = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True).transform
    return shapely_transform(project, geom)


def _reproject_box(geom, crs_from, crs_to):
    return _reproject_geom(geom, crs_from, crs_to)


def load_perimeters(shp_dir):
    """Load MTBS perimeters (EPSG:4269 → CONUS Albers EPSG:5070 for metric
    areas) and build an STRtree spatial index. Returns (gdf, tree)."""
    shp = os.path.join(shp_dir, "mtbs_perims_DD.shp")
    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(5070)  # all chips are CONUS; Albers keeps areas in m^2
    tree = STRtree(gdf.geometry.values)
    return gdf, tree


def parse_ig_date(value):
    """Parse MTBS Ig_Date ('2018-07-23', datetime64, or '20180723' ints)."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        s = str(int(value))
    else:
        s = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def _candidate_score(c):
    """Score = 0.5 x intersection_ratio + 0.5 x burn_mask_overlap (nan -> 0)."""
    o = c["burn_overlap"]
    if o is None or (isinstance(o, float) and np.isnan(o)):
        o = 0.0
    return W_INTER * c["inter_ratio"] + W_BURN_OVERLAP * o


def chip_candidates(merged_path, mask_path, perimeters_gdf, tree,
                    min_inter_ratio=MIN_INTERSECTION_RATIO,
                    max_gap=MAX_TEMPORAL_GAP_DAYS):
    """Collect and rank ALL spatial+temporal candidates for one chip.

    Returns (image_id, tile, date, footprint_wgs84_wkt, ranked) where
    `ranked` is a list of candidate dicts sorted by score desc, each with
    keys: idx, inter, inter_ratio, burn_overlap, gap, score, event_id,
    fire_name, incid_type, ig_date, post_id. `match_one_chip` keeps only
    the best candidate; the M1.6 ambiguous-case QC uses the top-2.
    """
    image_id = os.path.basename(merged_path).replace("_merged.tif", "")
    tile, date = parse_granule(merged_path)

    with rasterio.open(merged_path) as ds:
        crs = chip_crs_of(ds)
        b = ds.bounds
    footprint = box(b.left, b.bottom, b.right, b.top)
    footprint_mtbs = _reproject_box(footprint, crs, perimeters_gdf.crs)
    chip_area = footprint_mtbs.area
    # chip_bounds stored as WGS84 (portable across readers)
    footprint_wgs84 = _reproject_geom(footprint, crs, pyproj.CRS.from_epsg(4326))

    # Burn mask (chip CRS grid) for overlap scoring below.
    with rasterio.open(mask_path) as mds:
        mask = mds.read(1)
        mask_crs = chip_crs_of(mds)
        mask_transform = mds.transform
        mask_shape = (mds.height, mds.width)
    burned = mask == 1
    n_burned = int(burned.sum())

    hits = tree.query(footprint_mtbs)
    candidates = []
    for idx in hits:
        poly = perimeters_gdf.geometry.iloc[idx]
        inter = footprint_mtbs.intersection(poly).area
        if inter <= 0:
            continue
        inter_ratio = inter / chip_area
        if inter_ratio < min_inter_ratio:
            continue
        ig_date = parse_ig_date(perimeters_gdf["Ig_Date"].iloc[idx])
        if ig_date is None:
            gap = None
        else:
            gap = (date - ig_date).days
        if gap is not None and (gap < 0 or gap > max_gap):
            continue
        if n_burned == 0:
            overlap = np.nan
        else:
            # Rasterize the perimeter onto the chip grid, then measure the
            # fraction of burned pixels inside it (exact same grid, no resampling).
            poly_chip = _reproject_geom(poly, perimeters_gdf.crs, mask_crs)
            inside = rasterize([(poly_chip, 1)], out_shape=mask_shape,
                               transform=mask_transform, fill=0, dtype="uint8")
            overlap = float((inside == 1)[burned].mean())
        row = perimeters_gdf.iloc[idx]
        cand = {
            "idx": idx,
            "inter": inter,
            "inter_ratio": inter_ratio,
            "burn_overlap": overlap,
            "gap": gap,
            "event_id": str(row.get("Event_ID", "")),
            "fire_name": str(row.get("Incid_Name", "")),
            "incid_type": str(row.get("Incid_Type", "")),
            "ig_date": ig_date,
            "post_id": str(row.get("Post_ID", "") or ""),
        }
        cand["score"] = _candidate_score(cand)
        candidates.append(cand)

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return image_id, tile, date, footprint_wgs84.wkt, ranked


def match_one_chip(merged_path, mask_path, perimeters_gdf, tree,
                   min_inter_ratio=MIN_INTERSECTION_RATIO,
                   max_gap=MAX_TEMPORAL_GAP_DAYS):
    """Score all MTBS perimeter candidates for one chip; keep the best.

    Returns a dict with the 14-field schema (match_status/event_id still
    provisional: 'candidate' rows carry the best candidate's attributes;
    classification into matched/ambiguous/unmatched happens afterwards,
    with thresholds finalized from the full 790-candidate distribution).
    """
    image_id, tile, date, footprint_wkt, ranked = chip_candidates(
        merged_path, mask_path, perimeters_gdf, tree,
        min_inter_ratio=min_inter_ratio, max_gap=max_gap)

    if not ranked:
        return {
            "image_id": image_id, "hls_tile": tile,
            "acquisition_date": date.isoformat(),
            "chip_bounds": footprint_wkt, "mtbs_id": None,
            "fire_name": None, "fire_year": None,
            "intersection_area": None, "intersection_ratio": None,
            "burn_mask_overlap": None, "temporal_gap_days": None,
            "match_status": "unmatched", "match_confidence": 0.0,
            "second_score_margin": None, "post_id_prefix_match": None,
            "event_id": None,
        }

    best = ranked[0]
    best_row = perimeters_gdf.iloc[best["idx"]]
    best_score = best["score"]
    if len(ranked) > 1:
        second_score_margin = best_score - ranked[1]["score"]
    else:
        second_score_margin = 1.0  # no competing candidate → no ambiguity
    best_ig = parse_ig_date(best_row["Ig_Date"])
    year = None
    if "Year" in perimeters_gdf.columns and not pd.isna(best_row["Year"]):
        year = int(best_row["Year"])
    elif best_ig is not None:
        year = best_ig.year
    # Cross-check against the MTBS post-fire scene ID. MTBS Post_ID uses raw
    # Sentinel-2 granule names (e.g. 'A18SUD20210621_30m'), not HLS IDs, so the
    # independent signal is tile + acquisition-date coincidence.
    post_id = str(best_row.get("Post_ID", "") or "")
    post_prefix_match = None
    if post_id and post_id.lower() != "nan":
        post_prefix_match = (tile in post_id) and (
            date.strftime("%Y%m%d") in post_id)

    return {
        "image_id": image_id, "hls_tile": tile,
        "acquisition_date": date.isoformat(),
        "chip_bounds": footprint_wkt,
        "mtbs_id": str(best_row.get("Event_ID", "")),
        "fire_name": str(best_row.get("Incid_Name", "")),
        "fire_year": year,
        "second_score_margin": second_score_margin,
        "intersection_area": best["inter"],
        "intersection_ratio": best["inter_ratio"],
        "burn_mask_overlap": best["burn_overlap"],
        "temporal_gap_days": best["gap"],
        "match_status": "candidate", "match_confidence": best_score,
        "post_id_prefix_match": post_prefix_match,
        "event_id": None,  # filled after classification
    }


def classify(df, min_confidence=0.4, min_inter_ratio=0.01,
             min_overlap=0.9, ambiguity_margin=0.10, ambiguous_min=0.2):
    """Classify candidate rows using thresholds chosen from the full-match
    distribution (recorded in the M1 report).

    The burn_mask_overlap distribution was nearly binary over all 790
    candidate rows: 777 in [0.9, 1.0] (high-overlap candidate cluster),
    10 in [0, 0.1] (clear mismatches: multi-year temporal gaps, UNNAMED
    fires years before the chip), 3 in between. NOTE (M1.6 wording): the
    burn_mask_overlap >= 0.9 criterion marks a HIGH-OVERLAP CANDIDATE
    match, not an independent ground-truth verification — the MTBS
    perimeters are themselves the label source of hls_burn_scars.
    Thresholds:

    matched:   overlap >= min_overlap AND score >= min_confidence AND
               inter_ratio >= min_inter_ratio AND margin >= ambiguity_margin
    ambiguous: candidate, not matched, score >= ambiguous_min
    unmatched: everything else (candidates below ambiguous_min + rows with
               no candidates at all)
    """
    out = df.copy()
    cond_matched = (
        (out["match_status"] == "candidate")
        & (out["burn_mask_overlap"] >= min_overlap)
        & (out["match_confidence"] >= min_confidence)
        & (out["intersection_ratio"] >= min_inter_ratio)
        & (out["second_score_margin"] >= ambiguity_margin)
    )
    cond_ambiguous = (
        (out["match_status"] == "candidate")
        & ~cond_matched
        & (out["match_confidence"] >= ambiguous_min)
    )
    out.loc[cond_matched, "match_status"] = "matched"
    out.loc[cond_ambiguous, "match_status"] = "ambiguous"
    out.loc[out["match_status"] == "candidate", "match_status"] = "unmatched"
    out.loc[out["match_status"] == "matched", "event_id"] = out.loc[
        out["match_status"] == "matched", "mtbs_id"]
    return out


def iter_chips(root_dir, split_dir):
    """Yield (merged_path, mask_path) sorted by name."""
    pattern = os.path.join(root_dir, split_dir, "*_merged.tif")
    merged = sorted(glob.glob(pattern))
    for mp in merged:
        yield mp, mp.replace("_merged.tif", ".mask.tif")


def run_matching(chip_pairs, perimeters_gdf, tree):
    """Run match_one_chip over a list of (merged, mask) paths → DataFrame."""
    rows = []
    for merged, mask in chip_pairs:
        rows.append(match_one_chip(merged, mask, perimeters_gdf, tree))
    return pd.DataFrame(rows, columns=COLUMNS)

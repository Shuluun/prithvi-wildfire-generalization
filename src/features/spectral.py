"""Spectral baseline features: 6 HLS bands + NDVI + NBR.

Band order in the merged GeoTIFFs (1-based): 1 Blue B02, 2 Green B03,
3 Red B04, 4 NIR B8A, 5 SWIR1 B11, 6 SWIR2 B12 (reflectance scale).

NDVI = (NIR - Red) / (NIR + Red)
NBR  = (NIR - SWIR2) / (NIR + SWIR2)   (single-scene, no pre-fire dNBR in MVP)
"""
import hashlib

import numpy as np
import rasterio

FEATURE_NAMES = ["B02", "B03", "B04", "B8A", "B11", "B12", "NDVI", "NBR"]


def chip_rng(global_seed, image_id):
    """Deterministic per-chip RNG (global seed + image_id), so a chip samples
    the same pixels regardless of processing order or fold."""
    h = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8],
                       "big")
    return np.random.default_rng([global_seed, h])


def load_features(merged_path):
    """Return (X, mask) for one chip: X = (H*W, 8) float32 features in
    reflectance scale, mask = (H*W,) label array {-1, 0, 1}."""
    with rasterio.open(merged_path) as ds:
        img = ds.read().astype(np.float32)  # (6, H, W), reflectance
    h, w = img.shape[1], img.shape[2]
    blue, green, red = img[0], img[1], img[2]
    nir, swir1, swir2 = img[3], img[4], img[5]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)
    x = np.stack([blue, green, red, nir, swir1, swir2, ndvi, nbr], axis=-1)
    with rasterio.open(merged_path.replace("_merged.tif", ".mask.tif")) as mds:
        mask = mds.read(1).ravel()
    return x.reshape(-1, 8), mask


def sample_valid_indices(mask, cap, rng):
    """Deterministic UNIFORM sample (without replacement) of up to ``cap``
    valid pixel indices. Valid = label in {0, 1}; -1 (no-data/cloud) is
    excluded. The natural within-chip class balance is preserved — no
    per-class 50/50 forcing (M2 frozen sampling policy).

    Returns an int array of flat indices into the (H*W,) pixel axis.
    """
    valid = np.nonzero(mask >= 0)[0]
    if len(valid) <= cap:
        return valid
    return valid[rng.choice(len(valid), size=cap, replace=False)]


def valid_features_labels(x, mask):
    """Return (X_valid, y_valid) for ALL valid pixels of a chip (no sampling);
    used for threshold selection and evaluation. y in {0, 1}."""
    valid = mask >= 0
    return x[valid], mask[valid]


def sample_pixels(x, mask, n_per_class, rng, ignore_missing=True):
    """Deterministic stratified pixel sampling. Returns (X, y) with classes
    0 (unburned) and 1 (burned); -1 (missing) excluded."""
    idx_burn = np.nonzero(mask == 1)[0]
    idx_unburn = np.nonzero(mask == 0)[0]
    n_b = min(n_per_class, len(idx_burn))
    n_u = min(n_per_class, len(idx_unburn))
    sel_b = idx_burn[rng.choice(len(idx_burn), size=n_b, replace=False)]
    sel_u = idx_unburn[rng.choice(len(idx_unburn), size=n_u, replace=False)]
    sel = np.concatenate([sel_b, sel_u])
    return x[sel], mask[sel]


def burned_iou(y_true, y_pred):
    """Burned-class IoU (binary)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")

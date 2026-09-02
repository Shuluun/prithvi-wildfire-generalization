"""Frozen Prithvi-EO-2.0-300M encoder: build, preprocessing, feature extraction.

M4a/M4b use ONLY the foundation model ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M``.
The BurnScars downstream is referenced only for band semantics, reflectance-scale
preprocessing constants, and layer selection — it supplies NO trained downstream
weights. The backbone is built encoder-only and fully frozen.

Band order (== the HLS stack already validated in M2, ``src/features/spectral.py``):
BLUE (B02), GREEN (B03), RED (B04), NIR_NARROW (B8A), SWIR_1 (B11), SWIR_2 (B12).
"""
import hashlib

import numpy as np
import rasterio
import torch

from terratorch.datasets.utils import HLSBands

# --------------------------------------------------------------------------
# Frozen reflectance-scale (0-1) preprocessing constants — BurnScars downstream,
# copied from docs/prithvi_notes.md §4. NOT the DN-scale foundation config.
# --------------------------------------------------------------------------
BAND_ORDER = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
REFLECTANCE_MEAN = np.array(
    [0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
     0.2323245113436119, 0.1972854853760658, 0.11944914225186566],
    dtype=np.float32,
)
REFLECTANCE_STD = np.array(
    [0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
     0.07791732423672691, 0.08708738838140137, 0.07241979477437814],
    dtype=np.float32,
)

SELECTED_LAYERS = [5, 11, 17, 23]
BACKBONE_NAME = "prithvi_eo_v2_300"
MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
CHECKPOINT_FILE = "Prithvi_EO_V2_300M.pt"


def prithvi_bands():
    """The six HLS bands in the frozen semantic order (== merged.tif order)."""
    return [HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
            HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2]


def preprocessing_version():
    """Deterministic hash of the preprocessing identity (band order + mean/std),
    recorded in every cached sample so cached features are never silently reused
    against a different normalization."""
    h = hashlib.sha256()
    h.update("|".join(BAND_ORDER).encode())
    h.update(REFLECTANCE_MEAN.tobytes())
    h.update(REFLECTANCE_STD.tobytes())
    return h.hexdigest()[:16]


def build_backbone(device=None):
    """Build the encoder-only, fully-frozen Prithvi-EO-2.0-300M backbone."""
    from terratorch.registry import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        BACKBONE_NAME,
        pretrained=True,
        num_frames=1,
        bands=prithvi_bands(),
        features_only=True,  # encoder-only (PrithviViT), drops MAE decoder
    )
    # BACKBONE_REGISTRY returns a TimmBackboneWrapper; unwrap to the raw
    # PrithviViT so callers can use forward_features / blocks / embed_dim.
    while hasattr(model, "_timm_module"):
        model = model._timm_module
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    return model


def load_chip_bands_and_mask(merged_path):
    """Return (img, mask) for one chip.

    img: (6, H, W) float32 reflectance (band order == BAND_ORDER).
    mask: (H, W) int8 labels {-1, 0, 1}.
    """
    with rasterio.open(merged_path) as ds:
        img = ds.read().astype(np.float32)  # (6, H, W)
    with rasterio.open(merged_path.replace("_merged.tif", ".mask.tif")) as mds:
        mask = mds.read(1).astype(np.int8)
    return img, mask


def normalize_chip(img):
    """Z-score the six reflectance bands with the frozen BurnScars constants."""
    mean = REFLECTANCE_MEAN.reshape(-1, 1, 1)
    std = REFLECTANCE_STD.reshape(-1, 1, 1)
    return (img - mean) / std


def to_input_tensor(img, device=None):
    """(6, H, W) float32 -> (1, 6, H, W) float32 tensor (time dim added by model)."""
    t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)
    return t.to(device) if device is not None else t


def extract_layer_features(model, x):
    """Run the frozen encoder and return the selected layers as spatial maps.

    Returns a dict {layer_index: tensor (1, 1024, H', W')} with the CLS token
    removed and patch tokens reshaped to their (square) spatial grid. H', W' are
    derived from the actual token count (not hard-coded), so native 512x512
    yields 32x32 maps.
    """
    feats = model.forward_features(x)  # list of 24 tensors [1, tokens+1, 1024]
    out = {}
    for li in SELECTED_LAYERS:
        f = feats[li][:, 1:, :]  # drop CLS token
        b, n, c = f.shape
        h = w = int(round(n ** 0.5))
        if h * w != n:
            raise ValueError(f"layer {li}: {n} tokens is not a square grid")
        out[li] = f.reshape(b, c, h, w).contiguous()
    return out

"""M4a frozen-Prithvi feasibility tests.

Covers the pieces testable without loading the 300M backbone on GPU: band-order
semantics, reflectance-scale preprocessing constants, deterministic preprocessing
version, normalization math, input-tensor packing, the CLS-drop / square-grid
reshape logic in ``extract_layer_features``, and the frozen feature-cache
metadata/index (skipped when the cache has not been generated).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import RESULTS_ROOT  # noqa: E402
from src.models import prithvi  # noqa: E402


# --- band semantics + reflectance constants ----------------------------------

def test_band_order_matches_hls_stack():
    assert prithvi.BAND_ORDER == ["BLUE", "GREEN", "RED", "NIR_NARROW",
                                  "SWIR_1", "SWIR_2"]


def test_prithvi_bands_strings_match_order():
    assert [b.value for b in prithvi.prithvi_bands()] == prithvi.BAND_ORDER


def test_reflectance_constants_shape_and_scale():
    mean, std = prithvi.REFLECTANCE_MEAN, prithvi.REFLECTANCE_STD
    assert mean.shape == (6,) and std.shape == (6,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std > 0).all()
    # reflectance scale (0-1): means must live inside the unit interval
    assert (mean > 0).all() and (mean < 1).all()


def test_preprocessing_version_deterministic():
    a = prithvi.preprocessing_version()
    b = prithvi.preprocessing_version()
    assert a == b
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_preprocessing_version_sensitive_to_constants():
    base = prithvi.preprocessing_version()
    orig = prithvi.REFLECTANCE_MEAN.copy()
    prithvi.REFLECTANCE_MEAN = orig + 0.01
    try:
        assert prithvi.preprocessing_version() != base
    finally:
        prithvi.REFLECTANCE_MEAN = orig


# --- normalization / input packing -------------------------------------------

def test_normalize_chip_is_zscore():
    img = np.array([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]], [[0.7, 0.8]],
                    [[0.9, 1.0]], [[0.2, 0.3]]], dtype=np.float32)  # (6,1,2)
    out = prithvi.normalize_chip(img)
    mean = prithvi.REFLECTANCE_MEAN.reshape(-1, 1, 1)
    std = prithvi.REFLECTANCE_STD.reshape(-1, 1, 1)
    np.testing.assert_allclose(out, (img - mean) / std, rtol=1e-6)


def test_to_input_tensor_shape_and_dtype():
    img = np.zeros((6, 32, 32), dtype=np.float32)
    t = prithvi.to_input_tensor(img)
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (1, 6, 32, 32)
    assert t.dtype == torch.float32


# --- CLS-drop / square-grid reshape logic ------------------------------------

def _fake_model(n_tokens_per_block=1025, c=1024):
    """A stub exposing forward_features() -> list of 24 [1, tokens, C] tensors,
    standing in for PrithviViT.forward_features (list of block outputs)."""
    class _M:
        def forward_features(self, x):
            return [torch.zeros(1, n_tokens_per_block, c) for _ in range(24)]
    return _M()


def test_extract_layer_features_shapes():
    feats = prithvi.extract_layer_features(_fake_model(), None)
    assert set(feats.keys()) == set(prithvi.SELECTED_LAYERS)
    for li in prithvi.SELECTED_LAYERS:
        assert tuple(feats[li].shape) == (1, 1024, 32, 32)


def test_extract_layer_features_drops_cls():
    model = _fake_model()
    # tag the CLS token so we can verify it is removed from the reshaped maps
    raw = [torch.full((1, 1025, 1024), 7.0) for _ in range(24)]
    model.forward_features = lambda x: raw
    feats = prithvi.extract_layer_features(model, None)
    assert torch.all(feats[5] == 7.0)


def test_extract_layer_features_nonsquare_raises():
    # 9 tokens (3x3) + CLS would be square; 10 tokens + CLS (11 total -> 10
    # patch tokens) is NOT square and must raise.
    with pytest.raises(ValueError):
        prithvi.extract_layer_features(_fake_model(n_tokens_per_block=11), None)


# --- frozen feature-cache metadata (skip if not yet generated) ---------------

@pytest.fixture(scope="module")
def cache_dir():
    d = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache")
    if not os.path.exists(os.path.join(d, "m4a_cache_metadata.json")):
        pytest.skip("M4a feature cache not generated")
    return d


def test_cache_metadata(cache_dir):
    import json
    with open(os.path.join(cache_dir, "m4a_cache_metadata.json"),
              encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["model_id"] == "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    assert meta["backbone"] == "prithvi_eo_v2_300"
    assert meta["selected_layers"] == [5, 11, 17, 23]
    assert meta["input_band_order"] == prithvi.BAND_ORDER
    assert meta["feature_dtype"] == "float16"
    assert meta["n_events"] == 576
    assert meta["preprocessing_version"] == prithvi.preprocessing_version()


def test_cache_index_shape(cache_dir):
    idx = pd.read_csv(os.path.join(cache_dir, "m4a_cache_index.csv"))
    assert len(idx) == 576
    assert idx["feature_dtype"].eq("float16").all()
    assert idx["feature_shape"].eq("[4, 1024, 32, 32]").all()
    assert idx["token_grid_shape"].eq("[32, 32]").all()
    assert idx["image_id"].is_unique

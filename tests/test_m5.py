"""M5 decoder-comparison tests.

Covers the pieces testable without training: the three head architectures'
shapes, parameter counts (and the <=2M decoder budget), and — for M5a — the
constraint that the pointwise MLP uses NO 3x3 convolutions. The model-matrix
consistency test is skipped until the M5 scripts have run.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import RESULTS_ROOT  # noqa: E402
from src.m5.models import PointwiseMLP, SpatialDecoder, SpectralCNN  # noqa: E402


# --- M5a: pointwise MLP -----------------------------------------------------

def test_pointwise_mlp_has_no_spatial_convs():
    m = PointwiseMLP()
    for name, layer in m.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            assert layer.kernel_size == (1, 1), \
                f"{name} uses kernel {layer.kernel_size}, not pointwise 1x1"


def test_pointwise_mlp_param_count():
    m = PointwiseMLP()
    # 4096*256+256 + 256*64+64 + 64*1+1
    assert m.n_trainable == 1_065_345


def test_pointwise_mlp_forward_shape():
    m = PointwiseMLP()
    x = torch.zeros(1, 4096, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512)


def test_pointwise_mlp_dropout_inactive_in_eval():
    m = PointwiseMLP().eval()
    x = torch.randn(1, 4096, 32, 32)
    with torch.no_grad():
        a = m(x)
        b = m(x)
    assert torch.equal(a, b)


# --- M5b: lightweight spatial decoder ---------------------------------------

def test_spatial_decoder_param_budget():
    m = SpatialDecoder()
    assert 100_000 <= m.n_trainable <= 2_000_000


def test_spatial_decoder_forward_shape():
    m = SpatialDecoder()
    x = torch.zeros(1, 4, 1024, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512)


def test_spatial_decoder_has_spatial_convs():
    m = SpatialDecoder()
    kernels = {layer.kernel_size for name, layer in m.named_modules()
               if isinstance(layer, torch.nn.Conv2d)}
    assert (3, 3) in kernels  # it must actually do spatial context


# --- M5 matched spectral CNN control ----------------------------------------

def test_spectral_cnn_param_comparable_to_decoder():
    dec = SpatialDecoder().n_trainable
    cnn = SpectralCNN().n_trainable
    # same order of magnitude as the Prithvi decoder (matched capacity)
    assert 0.1 * dec <= cnn <= 10 * dec
    assert 100_000 <= cnn <= 2_000_000


def test_spectral_cnn_forward_shape():
    m = SpectralCNN()
    x = torch.zeros(1, 8, 512, 512)
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 1, 512, 512)


# --- model-matrix consistency (skip until M5 has run) -----------------------

@pytest.fixture(scope="module")
def model_matrix():
    path = os.path.join(RESULTS_ROOT, "m5_compare", "m5_model_matrix.csv")
    if not os.path.exists(path):
        pytest.skip("M5 comparison not yet generated")
    return pd.read_csv(path)


def test_model_matrix_has_all_five_models(model_matrix):
    # rf keeps the base column name (iou); the other four get _{key} suffixes
    for col in ["iou", "iou_linear", "iou_mlp", "iou_decoder", "iou_spectral"]:
        assert col in model_matrix.columns
        assert model_matrix[col].notna().sum() == 576


def test_model_matrix_events_match_primary():
    attrs = pd.read_csv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "metadata", "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"]
    path = os.path.join(RESULTS_ROOT, "m5_compare", "m5_model_matrix.csv")
    if not os.path.exists(path):
        pytest.skip("M5 comparison not yet generated")
    matrix = pd.read_csv(path)
    assert set(matrix["event_id"]) == set(primary["event_id"])

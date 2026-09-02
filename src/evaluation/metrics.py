"""Per-chip burned-area segmentation metrics.

All metrics are computed on valid pixels only (label -1 = no-data/cloud is
excluded by the caller before these functions are used). y_true in {0, 1};
y_prob in [0, 1] is the burned-class probability; a threshold turns it into a
binary prediction y_pred = (y_prob >= threshold).

Primary per-event metrics: burned IoU, Dice/F1. Secondary: precision, recall,
AUPRC, AUROC. Pixels are NOT independent statistical units — all inference is
aggregated at event level elsewhere (src/evaluation/bootstrap.py).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _cm(y_true, y_pred):
    """Confusion-matrix components for the burned (positive) class."""
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    return tp, fp, fn


def burned_iou(y_true, y_pred):
    tp, fp, fn = _cm(y_true, y_pred)
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")


def burned_dice(y_true, y_pred):
    """Burned-class Dice coefficient == F1 (binary)."""
    tp, fp, fn = _cm(y_true, y_pred)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else float("nan")


def precision(y_true, y_pred):
    tp, fp, _ = _cm(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom else float("nan")


def recall(y_true, y_pred):
    tp, _, fn = _cm(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else float("nan")


def _auc(kind, y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return float("nan")  # both classes required for a ranked metric
    try:
        if kind == "auprc":
            return float(average_precision_score(y_true, y_prob))
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def per_chip_metrics(y_true, y_prob, threshold):
    """All metrics for one chip's valid pixels at a fixed threshold."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "iou": burned_iou(y_true, y_pred),
        "dice": burned_dice(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "auprc": _auc("auprc", y_true, y_prob),
        "auroc": _auc("auroc", y_true, y_prob),
        "n_valid": int(len(y_true)),
        "true_burn_fraction": float(y_true.mean()),
        "pred_burn_fraction": float(y_pred.mean()),
    }


def select_threshold(y_true, y_prob, grid):
    """Pick the threshold maximizing pooled burned Dice/F1 over a fixed grid.

    Returns (best_threshold, best_dice). Deterministic (ties go to the first
    threshold scanned).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best_t, best_d = float(grid[0]), -np.inf
    for t in grid:
        d = burned_dice(y_true, (y_prob >= t).astype(np.int64))
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, best_d

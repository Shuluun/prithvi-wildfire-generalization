"""Diagnostic: are the cached frozen Prithvi features informative for burn-scar?

Independent of the torch LinearProbe training loop — trains a scikit-learn
LogisticRegression (LBFGS) on the 4096-dim concatenated layer features at the
32x32 token grid, with the block-majority label, and reports AUROC/IoU/Dice.
If sklearn also returns ~chance AUROC, the features (or their alignment) are at
fault. If sklearn performs well, the torch probe training loop is at fault.
"""
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402

FEAT_DIR = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache", "features")


def block_labels(mask):
    """Downsample a (512,512) mask to (32,32) block-majority labels {-1,0,1}."""
    y = np.zeros((32, 32), np.int8)
    for i in range(32):
        for j in range(32):
            blk = mask[16 * i:16 * i + 16, 16 * j:16 * j + 16]
            valid = blk[blk >= 0]
            y[i, j] = int(valid.mean() > 0.5) if len(valid) else -1
    return y


def main():
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"]
    split = dict(zip(primary["image_id"], primary["split"]))

    X, Y = [], []
    for image_id in list(primary["image_id"])[:8]:
        a = np.load(os.path.join(FEAT_DIR, f"{image_id}.npy")).astype(np.float32)
        a = a.reshape(4096, 32, 32)  # (C, 32, 32)
        mp = os.path.join(HLS_BURN_SCARS_DIR, split[image_id],
                          image_id + "_merged.tif")
        with rasterio.open(mp.replace("_merged.tif", ".mask.tif")) as ds:
            mask = ds.read(1).astype(np.int8)
        yb = block_labels(mask)
        valid = yb >= 0
        X.append(a[:, valid].T)          # (n_valid_blocks, 4096)
        Y.append(yb[valid].astype(np.int64))

    X = np.concatenate(X, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0)
    print(f"blocks: {X.shape}, pos fraction: {Y.mean():.3f}")

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                             solver="lbfgs", n_jobs=-1)
    clf.fit(X, Y)
    p = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(Y, p)
    apr = average_precision_score(Y, p)
    print(f"in-sample  AUROC={auc:.4f}  AUPRC={apr:.4f}")

    # a held-out chip for a fairer read
    a = np.load(os.path.join(FEAT_DIR, f"{primary['image_id'].iloc[9]}.npy")).astype(np.float32).reshape(4096, 32, 32)
    mp = os.path.join(HLS_BURN_SCARS_DIR, split[primary['image_id'].iloc[9]],
                      primary['image_id'].iloc[9] + "_merged.tif")
    with rasterio.open(mp.replace("_merged.tif", ".mask.tif")) as ds:
        mask = ds.read(1).astype(np.int8)
    yb = block_labels(mask); valid = yb >= 0
    Xh = a[:, valid].T.astype(np.float32); Yh = yb[valid].astype(np.int64)
    ph = clf.predict_proba(Xh)[:, 1]
    print(f"held-out chip AUROC={roc_auc_score(Yh, ph):.4f} AUPRC={average_precision_score(Yh, ph):.4f}")


if __name__ == "__main__":
    main()

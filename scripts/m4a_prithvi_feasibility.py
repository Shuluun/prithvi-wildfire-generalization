"""M4a — frozen Prithvi-EO-2.0-300M encoder feasibility.

Subcommands:
  verify   load the foundation backbone, verify architecture + full freeze,
           and empirically confirm native 512x512 inference (dynamic sincos
           positional embedding; no resize, no pos-embed adaptation).
  smoke    preprocessing sanity checks + encoder smoke test on 5-10 real chips.
  extract  cache frozen [5,11,17,23] layer representations for all 576 primary
           Wildfire events (float16) on D:.
  sanity   numerical cache sanity: re-extract a sample and compare to cache.

Usage (from repo root):
  python scripts/m4a_prithvi_feasibility.py verify
  python scripts/m4a_prithvi_feasibility.py smoke
  python scripts/m4a_prithvi_feasibility.py extract
  python scripts/m4a_prithvi_feasibility.py sanity
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import HLS_BURN_SCARS_DIR, METADATA_ROOT, RESULTS_ROOT  # noqa: E402
from src.models import prithvi  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_ROOT = os.path.join(RESULTS_ROOT, "m4a", "prithvi_cache")


# --------------------------------------------------------------------------
# verify — step 1
# --------------------------------------------------------------------------
def cmd_verify():
    print("=== M4a step 1: foundation-model loading + freeze + native 512 ===")
    model = prithvi.build_backbone(device=DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"backbone type     : {type(model).__name__}")
    print(f"num_blocks (depth): {len(model.blocks)}")
    print(f"embed_dim         : {model.embed_dim}")
    print(f"patch_size        : {tuple(model.patch_embed.patch_size)}")
    print(f"patch_embed input_size: {tuple(model.patch_embed.input_size)}")
    print(f"patch_embed grid_size : {tuple(model.patch_embed.grid_size)}")
    print(f"num_patches       : {model.patch_embed.num_patches}")
    print(f"num_frames        : {model.num_frames}")
    print(f"in_chans          : {model.patch_embed.proj.in_channels}")
    print(f"parameters        : {n_params:,}  trainable={n_trainable:,}")

    assert type(model).__name__ == "PrithviViT"
    assert len(model.blocks) == 24
    assert model.embed_dim == 1024
    assert tuple(model.patch_embed.patch_size) == (1, 16, 16)
    assert n_trainable == 0, "backbone must be fully frozen"
    assert all(not p.requires_grad for p in model.parameters())
    print("freeze check      : OK (requires_grad=False for every parameter)")

    # Empirically derive the token grid for 224 vs 512 (do not assume it).
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
    for size in (224, 512):
        x = torch.zeros(1, 6, size, size, device=DEVICE)
        t0 = time.time()
        with torch.no_grad():
            feats = model.forward_features(x)
        dt = time.time() - t0
        grids = {}
        for li in prithvi.SELECTED_LAYERS:
            f = feats[li]
            tok = f.shape[1]
            grids[li] = (int((tok - 1) ** 0.5), int((tok - 1) ** 0.5),
                         f.shape[-1], f.dtype)
        print(f"  input {size}x{size}: layers {prithvi.SELECTED_LAYERS} "
              f"-> token+CLS grid {grids}  t={dt:.3f}s")
    if DEVICE == "cuda":
        print(f"  GPU peak mem: {torch.cuda.max_memory_allocated()/2**20:.1f} MB")
    print("verify: PASS")


# --------------------------------------------------------------------------
# smoke — steps 2 & 3
# --------------------------------------------------------------------------
def cmd_smoke():
    print("=== M4a steps 2-3: preprocessing sanity + encoder smoke ===")
    model = prithvi.build_backbone(device=DEVICE)
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))

    # pick representative chips by burned fraction (low / median / high)
    s = primary.sort_values("burned_fraction").reset_index(drop=True)
    n = len(s)
    picks = pd.concat([s.iloc[:2], s.iloc[[n // 2]], s.iloc[[n - 2]]])
    picks = picks.drop_duplicates("image_id")
    print(f"chips selected: {len(picks)} (low/median/high burned fraction)")

    rows = []
    for _, r in picks.iterrows():
        image_id = r["image_id"]
        mp = os.path.join(HLS_BURN_SCARS_DIR, img_split[image_id],
                          image_id + "_merged.tif")
        img, mask = prithvi.load_chip_bands_and_mask(mp)
        raw_min, raw_max = img.min(), img.max()
        assert np.isfinite(img).all(), "raw reflectance has NaN/Inf"
        norm = prithvi.normalize_chip(img)
        assert np.isfinite(norm).all(), "normalized has NaN/Inf"
        assert img.shape[0] == 6, "expected 6 bands"

        x = prithvi.to_input_tensor(norm, device=DEVICE)
        t0 = time.time()
        with torch.no_grad():
            feats = prithvi.extract_layer_features(model, x)
        dt = time.time() - t0
        info = {
            "image_id": image_id,
            "event_id": r["event_id"],
            "burned_fraction": float(r["burned_fraction"]),
            "input_shape": list(x.shape),
            "n_nodata": int((mask == -1).sum()),
            "t_sec": dt,
        }
        for li in prithvi.SELECTED_LAYERS:
            f = feats[li]
            info[f"layer{li}_shape"] = list(f.shape)
        rows.append(info)
        print(f"  {image_id[:36]} burn={r['burned_fraction']:.3f} "
              f"in{x.shape[1:]} nodata={info['n_nodata']} "
              f"raw[{raw_min:.3f},{raw_max:.3f}] "
              f"L5={list(feats[5].shape)} L23={list(feats[23].shape)} "
              f"t={dt:.3f}s")

    out_dir = os.path.join(RESULTS_ROOT, "m4a")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "m4a_smoke.csv"), index=False)
    print(f"-> {os.path.join(out_dir, 'm4a_smoke.csv')}")
    print("smoke: PASS")


# --------------------------------------------------------------------------
# extract — step 4 (cache all 576 events)
# --------------------------------------------------------------------------
def cmd_extract():
    print("=== M4a step 4: frozen feature cache (576 events) ===")
    model = prithvi.build_backbone(device=DEVICE)
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"].copy()
    img_split = dict(zip(primary["image_id"], primary["split"]))

    # size estimate: 4 layers x 1024 x 32 x 32 x 2 bytes (fp16) per 512 chip
    per_chip = 4 * 1024 * 32 * 32 * 2
    total = per_chip * len(primary)
    print(f"estimated cache size: {per_chip/2**20:.1f} MB/chip x "
          f"{len(primary)} = {total/2**30:.2f} GB (fp16)")

    feat_dir = os.path.join(CACHE_ROOT, "features")
    os.makedirs(feat_dir, exist_ok=True)
    meta = {
        "model_id": prithvi.MODEL_ID,
        "backbone": prithvi.BACKBONE_NAME,
        "checkpoint": prithvi.CHECKPOINT_FILE,
        "selected_layers": prithvi.SELECTED_LAYERS,
        "input_band_order": prithvi.BAND_ORDER,
        "preprocessing_version": prithvi.preprocessing_version(),
        "feature_dtype": "float16",
        "num_frames": 1,
        "n_events": int(len(primary)),
    }
    rows = []
    t_start = time.time()
    for i, (_, r) in enumerate(primary.iterrows()):
        image_id = r["image_id"]
        mp = os.path.join(HLS_BURN_SCARS_DIR, img_split[image_id],
                          image_id + "_merged.tif")
        img, mask = prithvi.load_chip_bands_and_mask(mp)
        norm = prithvi.normalize_chip(img)
        x = prithvi.to_input_tensor(norm, device=DEVICE)
        with torch.no_grad():
            feats = prithvi.extract_layer_features(model, x)
        stacked = np.stack([feats[li][0].cpu().numpy().astype(np.float16)
                            for li in prithvi.SELECTED_LAYERS], axis=0)
        # stacked: (4, 1024, H', W') fp16
        fpath = os.path.join(feat_dir, f"{image_id}.npy")
        np.save(fpath, stacked)
        h, w = stacked.shape[2], stacked.shape[3]
        rows.append({
            "image_id": image_id, "event_id": r["event_id"],
            "token_grid_shape": [h, w],
            "feature_shape": list(stacked.shape),
            "feature_dtype": "float16",
        })
        if (i + 1) % 50 == 0:
            print(f"  cached {i+1}/{len(primary)}  "
                  f"({(time.time()-t_start)/60:.1f} min elapsed)")

    with open(os.path.join(CACHE_ROOT, "m4a_cache_metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(CACHE_ROOT, "m4a_cache_index.csv"),
                              index=False)
    print(f"extract: cached {len(rows)} events in {feat_dir}")
    print(f"total wall-clock: {(time.time()-t_start)/60:.1f} min")


# --------------------------------------------------------------------------
# sanity — step 5
# --------------------------------------------------------------------------
def cmd_sanity(n=4):
    print(f"=== M4a step 5: numerical cache sanity ({n} chips) ===")
    model = prithvi.build_backbone(device=DEVICE)
    attrs = pd.read_csv(os.path.join(METADATA_ROOT, "event_attributes.csv"))
    primary = attrs[attrs["incid_type"] == "Wildfire"]
    img_split = dict(zip(primary["image_id"], primary["split"]))
    feat_dir = os.path.join(CACHE_ROOT, "features")

    sample = primary.sample(n=n, random_state=0)
    max_abs_diff = 0.0
    for _, r in sample.iterrows():
        image_id = r["image_id"]
        cached = np.load(os.path.join(feat_dir, f"{image_id}.npy"))  # fp16
        mp = os.path.join(HLS_BURN_SCARS_DIR, img_split[image_id],
                          image_id + "_merged.tif")
        img, _ = prithvi.load_chip_bands_and_mask(mp)
        norm = prithvi.normalize_chip(img)
        x = prithvi.to_input_tensor(norm, device=DEVICE)
        with torch.no_grad():
            feats = prithvi.extract_layer_features(model, x)
        fresh = np.stack([feats[li][0].cpu().numpy().astype(np.float32)
                          for li in prithvi.SELECTED_LAYERS], axis=0)
        assert cached.shape == fresh.shape, (cached.shape, fresh.shape)
        assert np.isfinite(cached).all(), "cached has non-finite values"
        d = float(np.max(np.abs(cached.astype(np.float32) - fresh)))
        max_abs_diff = max(max_abs_diff, d)
        print(f"  {image_id[:40]} shape={cached.shape} max_abs_diff={d:.4e}")
    print(f"sanity: PASS (max abs diff {max_abs_diff:.4e}, fp16 rounding-level)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "smoke", "extract", "sanity"])
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    {"verify": cmd_verify, "smoke": cmd_smoke, "extract": cmd_extract,
     "sanity": lambda: cmd_sanity(args.n)}[args.command]()

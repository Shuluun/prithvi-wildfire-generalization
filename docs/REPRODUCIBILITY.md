# Reproducibility

This document records every step needed to reproduce the study's inputs, splits, models, and
figures. It does **not** re-run the expensive experiments; it records what a clean checkout must
have and what each command produces.

## 1. Environment

- Interpreter: the `prithvi-wf` conda environment (Python 3.11.16).
- Key packages: `torch==2.5.0+cu124`, `terratorch==0.99.7`, `scikit-learn`, `rasterio`,
  `geopandas`, `matplotlib`, `pandas`, `numpy`.
- Full pinning: `environment.yml` (conda) and `requirements-lock.txt` (pip).

## 2. Paths (env-controlled)

- Raw data and caches live **outside** this repository. `src/data/paths.py` defaults the storage
  root to a sibling `data/` directory next to the repo's parent; set `PRITHVI_WF_ROOT` to point
  elsewhere.
- Raw data (NOT in git): `<root>/data/raw/hls_burn_scars/{train,val,test}/` (each chip
  `..._merged.tif` + `.mask.tif`) and `<root>/data/raw/mtbs/` (MTBS perimeters shapefile).
- Small tracked metadata (IN git): `data/metadata/` (events, event_attributes, chip_inventory,
  splits).
- Results (IN git, except regenerable prediction maps): `results/`.

## 3. Foundation checkpoint identity (recorded, not re-downloaded)

- Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`; backbone `prithvi_eo_v2_300`; checkpoint file
  `Prithvi_EO_V2_300M.pt` (foundation only, **no** BurnScars-finetuned weights). 303,886,336
  parameters, **frozen** (`requires_grad = False`).
- Frozen layers `[5, 11, 17, 23]`, CLS token dropped, patch tokens → `[B, 1024, 32, 32]` per
  layer; concat = `[B, 4096, 32, 32]`.
- Reflectance-scale (0–1) preprocessing constants are hard-recorded in `src/models/prithvi.py`
  (`REFLECTANCE_MEAN` / `REFLECTANCE_STD`), band order `BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`; a
  preprocessing hash (`preprocessing_version()`) guards the feature cache against silent mismatch.

## 4. Random seeds (recorded)

- Split seed: **42** (all `split_k5_*_seed42.csv`).
- Bootstrap seed: **42**, 10,000 resamples (`src/evaluation/bootstrap.py`).
- Per-chip pixel sampling: `chip_rng(SEED=42, image_id)`.
- Threshold grid: `linspace(0.01, 0.99, 99)`, maximized on inner-val pooled burned Dice.

## 5. Commands (run from repo root; `python` = the env interpreter above)

### Data preparation, event reconstruction, splits
```bash
python scripts/m1_chip_inventory.py                 # chip_inventory.csv
python scripts/m1_reconstruct_events.py --pilot 100 # pilot matching
python scripts/m1_reconstruct_events.py --full      # full MTBS matching
python scripts/m1_analyze_events.py                 # events.csv + per-event stats
python scripts/m1_5_qc.py                           # QC audit
python scripts/m1_5_visual_qc.py                    # visual QC images
python scripts/m1_5_make_splits.py                  # spatial-overlap / tile-disjoint splits
python scripts/m1_6_make_protocol.py                # shared event-disjoint K5 (primary)
python scripts/m1_6_ambiguous_qc.py                 # ambiguous-event QC
```

### Spectral baseline
```bash
python scripts/m2_make_inner_split.py               # inner K5 folds (seed 42)
python scripts/m2_spectral_baseline.py --protocol event # primary RF -> OOF IoU/Dice/AUROC
python scripts/m2_paired_sensitivity.py             # primary-vs-spatial paired deltas
python scripts/m2_make_figures.py                   # spectral-baseline figures
```

### Frozen-Prithvi feature extraction + linear probe
```bash
python scripts/m4a_prithvi_feasibility.py verify    # env/checkpoint sanity
python scripts/m4a_prithvi_feasibility.py smoke     # 4-chip smoke
python scripts/m4a_prithvi_feasibility.py extract   # 576-event frozen cache (results/m4a/prithvi_cache/)
python scripts/m4a_prithvi_feasibility.py sanity    # cache sanity
python scripts/m4b_linear_probe.py                  # frozen linear probe
```

### Representation diagnostics
```bash
python scripts/m4_5_diagnostics.py                  # error overlap, stratifications, layers
python scripts/m4_5_make_figures.py
```

### Nonlinear / spatial / matched-spectral readouts
```bash
python scripts/m5a_pointwise_mlp.py                 # pointwise MLP (no 3x3 conv)
python scripts/m5b_spatial_decoder.py               # lightweight spatial decoder
python scripts/m5_spectral_cnn_control.py           # matched spectral CNN control
python scripts/m5_compare.py                        # 5-model matrix + paired bootstrap CIs
```

`m5_compare.py` rebuilds the committed `results/m5_compare/` tables from the committed per-event
out-of-fold predictions and thresholds. Only the boundary-error table additionally requires the
raw HLS imagery and the regenerated `.npy` prediction maps (see §6).

### Final figures
```bash
python scripts/m6_make_figures.py                   # fig1..fig6 -> results/figures/final/
```

### Tests
```bash
python -m pytest tests/ -q
```

## 6. What is committed vs regenerable

| Artifact | Committed? | Note |
|---|---|---|
| `data/metadata/*.csv` (events, attrs, inventory, K5 splits) | ✅ yes | event linkage auditable in git |
| `data/metadata/splits/split_loeo_seed42.csv` (~55 MB) | ❌ no | regenerable via `m1_5_make_splits.py`; superseded by K5 manifests |
| per-event OOF predictions + thresholds (`results/{m2,m4_linear_probe,m5a,m5b,m5_spectral_cnn}/…`) | ✅ yes | primary per-event results; feed `m5_compare.py` |
| `results/m5_compare/*.csv` (matrix, ranking, deltas, …) | ✅ yes | frozen summary tables; **feed all final figures** |
| final figures `results/figures/final/*.png` | ✅ yes | generated by `m6_make_figures.py` |
| raw HLS/MTBS data, frozen feature cache, `.npy` prediction maps, `.log`, weights | ❌ no | gitignored; regenerable via the commands above |

## 7. Reproducibility audit checklist

- [x] environment spec pinned (`environment.yml`, `requirements-lock.txt`)
- [x] data acquisition/preparation command recorded (§5)
- [x] event reconstruction command recorded (§5)
- [x] split generation command recorded (§5)
- [x] spectral-baseline RF command recorded
- [x] feature extraction + linear probe commands recorded
- [x] readout-comparison commands recorded
- [x] figure-generation command recorded
- [x] random seeds recorded (§4)
- [x] paths env-controlled (`PRITHVI_WF_ROOT`); no hardcoded absolute paths
- [x] large caches/data/weights gitignored (`.gitignore`)
- [x] exact model checkpoint recorded (§3)
- [x] result CSVs needed for figures committed (§6)
- [x] raw large predictions not committed (regenerable)

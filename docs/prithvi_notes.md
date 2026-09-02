# Prithvi-EO-2.0 / BurnScars — Technical Investigation Notes

Sources: NASA-IMPACT/Prithvi-EO-2.0 repository, TerraTorch (IBM/torchgeo), HuggingFace model
cards, arXiv:2412.02732. Values below were read from these sources; anything not confirmed is
marked as such.

## 1. Two official reference pipelines (do NOT conflate)

| | `configs/firescars.yaml` (NASA repo) | `Prithvi-EO-2.0-300M-BurnScars` (HF) |
|---|---|---|
| Role | repo benchmark / example pipeline | official downstream model |
| Input | 224×224 (512 resized via albumentations) | 512×512 chip → 224 |
| Decoder | **UperNetDecoder** | **UNetDecoder** |
| Encoder features | single-scale (SelectIndices + ReshapeTokensToImage) | **multi-layer `[5, 11, 17, 23]`** |
| Data module | `terratorch.datamodules.FireScarsNonGeoDataModule` | `GenericNonGeoSegmentationDataModule` (confirmed from `burn_scars_config.yaml`) |
| Reported | — | test IoU(burned) 87.52, mIoU 93.00 |

This project studies **frozen-representation** cross-event generalization and uses the multi-layer
spatial-feature approach of the BurnScars pipeline; it does not reproduce full fine-tuning.

## 2. Model architecture — Prithvi-EO-2.0-300M

From the HF `config.json`:

- architecture `prithvi_eo_v2_300` (ViT-Large)
- `depth` 24, `embed_dim` 1024, `num_heads` 16, `in_chans` 6
- `img_size` 224, `patch_size` `[1, 16, 16]` (time=1, h=16, w=16)
- `num_frames` 4 (pretraining), `mask_ratio` 0.75 (MAE)

Note: the paper reports the 600M (ViT-H) patch as 14×14; the 300M (ViT-L) uses patch 16 per
`config.json`. Burn-scar fine-tuning uses **single-timestamp** (`num_frames=1`).

## 3. Input bands and order

Six bands, semantic order: **Blue, Green, Red, NIR-Narrow (B8A), SWIR1 (B11), SWIR2 (B12)**.

- `firescars.yaml` lists: `BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2`.
- Dataset card lists: Blue, Green, Red, Narrow NIR, SWIR, SWIR 2.
- The foundation `config.json` labels (`["B02".."B07"]`) must **not** be equated with the
  task-specific six-band semantic order. The actual tensor channel order must be validated against
  the task-specific preprocessing/data pipeline (band-order validation).

## 4. Normalization — two scales, do not mix

- **Foundation `config.json` (DN scale)** — confirmed values:
  - mean `[1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]`
  - std  `[2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]`
- **TerraTorch / BurnScars downstream preprocessing (reflectance 0–1 scale)** — z-score per band,
  read from `burn_scars_config.yaml` (HF `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`):
  - mean `[0.033349706741586264, 0.05701185520536176, 0.05889748132001316, 0.2323245113436119,
    0.1972854853760658, 0.11944914225186566]`
  - std  `[0.02269135568823774, 0.026807560223070237, 0.04004109844362779, 0.07791732423672691,
    0.08708738838140137, 0.07241979477437814]`
  - band order in that config (dataset & output): `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`;
    `rgb_indices: [2, 1, 0]`; `ignore_index: -1` (labels `-1` = no-data); `no_data_replace: 0`.
  - official BurnScars config: `backbone prithvi_eo_v2_300` (pretrained), necks
    `SelectIndices [5,11,17,23] → ReshapeTokensToImage → LearnedInterpolateToPyramidal`,
    `UNetDecoder` channels `[512,256,128,64]`, `freeze_backbone: false` (official fine-tunes; this
    project freezes), AdamW lr 1e-4, batch 8, seed 2, bf16-mixed.
  - official splits: `splits/{train,val,test}.txt` = **804 files** (524 / 160 / 120),
    `*_merged.tif` images with `*.mask.tif` labels (see §6 count discrepancy).

Implementation must record the data scale, verify reflectance vs DN, and add a preprocessing
sanity check (including band-order validation).

## 5. Frozen embedding extraction

- Build the backbone: `terratorch.registry.BACKBONE_REGISTRY.build("prithvi_eo_v2_300",
  pretrained=True, bands=[...], num_frames=1)`.
- Backbone forward output: list of tensors shaped `[batch, token, 1024]`, **including the CLS
  token**.
- Multi-scale spatial features for the 300M: intermediate layers `[5, 11, 17, 23]` (via the
  `SelectIndices` neck in TerraTorch).
- CLS token = global image-level embedding; patch tokens = spatial features. For pixel-wise
  segmentation, use the patch tokens (spatial), not only the CLS token.
- Import paths (verified on terratorch 0.99.7):
  - `PrithviViT`: `terratorch.models.backbones.prithvi_mae`
  - `PrithviModelFactory`: `terratorch.models.prithvi_model_factory`
  - `EncoderDecoderFactory`: `terratorch.models.encoder_decoder_factory`
    (`_get_backbone` → `BACKBONE_REGISTRY.build`, `_get_decoder_and_head_kwargs` →
    `DECODER_REGISTRY`, sources `terratorch` + `smp`)
  - `UperNetDecoder`: `terratorch.models.decoders.upernet_decoder` (terratorch source)
  - `UNetDecoder`: resolved via the **smp** source of `DECODER_REGISTRY`
    (`segmentation-models-pytorch`), not a terratorch class.

## 6. hls_burn_scars dataset

- 512×512 chips, 6 bands, labels `1`=burned / `0`=unburned / `-1`=no-data/cloud; CONUS 2018–2021;
  labels from MTBS burn-scar shapefiles; roughly 88% / 11% / 1% split.
- **Count discrepancy — resolved:** the loading script's TEST split generator re-points at the
  validation dir (`"data": val_data`), so the viewer's test split (264) duplicates validation →
  1068 rows over **804 unique scenes (540 train + 264 val, 0 cross-split overlap)**.
- **No explicit `event_id`** is shipped — event identity must be reconstructed via MTBS matching
  (see `src/data/event_reconstruction.py`). Reconstructed by `scripts/m1_reconstruct_events.py`:
  756/804 chips matched to unique MTBS events with real CRS/affine/bounds/date matching; **every
  event has exactly 1 chip** (structural: one post-fire scene per fire). 24 ambiguous + 24
  unmatched chips excluded by design.
- GeoTIFFs carry real georeferencing (UTM per tile, 30 m) and acquisition dates are parsed from the
  HLS granule ID (julian day).

## 7. TerraTorch dependency structure

`terratorch` pulls a heavy, pinned stack, including: `torch==2.5.0`, `pytorch-lightning==2.4.0`,
`timm==0.9.7`, `torchgeo==0.6.1`, `transformers`, and geospatial libraries (`rasterio`, `geopandas`,
`xarray`, `rioxarray`, `shapely`, `pyproj`). Pin the environment; Python 3.11 is required (3.13 is
not compatible with this stack).

**Version conflict in official materials:** the official `NASA-IMPACT/Prithvi-EO-2.0`
requirements.txt pins terratorch @ git commit `ca289f06` (= PyPI 0.99.5, 2024-11-26) — but that
commit **predates** the registration of `prithvi_eo_v2_300` (added 2024-11-27/28,
`terratorch/models/backbones/prithvi_mae.py`, commit "Fixed padding and updated prithvi names").
The repo's own `firescars.yaml` and the HF BurnScars config reference `prithvi_eo_v2_300`, so the
official pin is internally inconsistent. **Project decision:** use **PyPI `terratorch==0.99.7`**
(2024-12-06, BurnScars release era), which registers `prithvi_eo_v2_300` (via timm registry; see
§5) and is compatible with the official 2024-11 pin stack (`torch<=2.5.0`, `lightning 2.x != 2.3`,
`torchgeo>=0.6.0`, `torchmetrics<=1.3.1`). terratorch 0.99.8+ forces torch 2.6 / lightning 2.5 —
rejected. In 0.99.x, Prithvi backbones are registered into **timm**'s model registry
(`prithvi_eo_v2_300`, `prithvi_vit_100`, …); `BACKBONE_REGISTRY` dispatches names via a
`timm_`/`terratorch_` prefix (`MultiSourceRegistry`) but also falls back across sources, so
`BACKBONE_REGISTRY.build("prithvi_eo_v2_300", ...)` resolves without a prefix (or use
`timm.create_model("prithvi_eo_v2_300")`).

## 8. Event reconstruction and audit outcomes

- **MTBS source**: legacy ZipServlet (burnseverity portal) is dead (HTTP 503). Use USGS
  ScienceBase item `5e7229b8e4b01d509268afba` (Burned Areas Boundaries ver. 12.0, Apr 2025;
  release DOI 10.5066/P9IED7RZ), `mtbs_perims_DD.zip` 374,092,911 B, 30,730 perimeters,
  EPSG:4269 → EPSG:5070 for metric areas. Fields used: `Event_ID`, `Incid_Name`, `Incid_Type`,
  `Ig_Date`, `Post_ID` (raw Sentinel-2 granule names — NOT HLS IDs).
- **Matching worked at scale**: burn-mask overlap nearly binary (777/790 candidates ≥ 0.9; 10 ≈ 0
  with multi-year gaps). Thresholds (overlap ≥ 0.9, score ≥ 0.4, margin ≥ 0.1) chosen from the
  distribution, recorded in `src/data/event_reconstruction.py::classify`.
- **Split/leakage artifacts**: deterministic manifests in `data/metadata/splits/` (conventional
  chip-level seed-42 + 756-fold LOEO); leakage tests pass (`tests/test_splits.py`).
- **Protocol correction (2026-08-31)**: with 1 chip per event, the chip-level manifest is
  automatically event-disjoint and there is NO leak-free within-event protocol. The PRIMARY
  protocol is deterministic 5-fold event-disjoint CV (seed 42) over the 576 matched-Wildfire
  population (`split_k5_event_seed42.csv`, one shared assignment for both models), with a
  spatial-overlap-clean variant (`split_k5_spatial_seed42.csv`, zero cross-fold footprint overlap,
  proven) and an HLS-tile-disjoint variant (sensitivity only). See `src/data/splits.py`.
- **Structural limitation for the experiment phase**: 34.4% of LOEO folds have ≥1 training chip
  spatially overlapping the held-out chip (199 overlapping event pairs) — geographic memorization
  risk. Addressed by the spatial-overlap-grouped K5 protocol, which removes all 109
  primary-population overlapping pairs from the train/test boundary; residual risk is adjacent
  non-intersecting chips (partially controlled by the tile-disjoint variant).

## 9. Hardware notes

- ≈8 GB of VRAM is sufficient for frozen embedding extraction plus a lightweight decoder; full
  fine-tuning at 224×224 / batch 16 would likely exceed this memory.
- Raw data and caches are stored outside the repository (see `src/data/paths.py`).

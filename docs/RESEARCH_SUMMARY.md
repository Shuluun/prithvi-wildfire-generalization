# Research Summary — Cross-Event Wildfire Burn-Scar Segmentation with Frozen Prithvi-EO-2.0

**Date:** 2026-09-02

This document is the complete research narrative. Every number reproduces a committed result CSV
(see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)); no new results are introduced here.

---

## 1. Research question

> Do **frozen Prithvi-EO-2.0 representations** transfer better than **conventional
> spectral representations** for burned-area segmentation on **unseen wildfire events**?

The answer from the current experiments is **negative**, with conservative wording. The preferred
conclusion:

> **Frozen Prithvi-EO-2.0 representations transferred poorly to cross-event single-date
> wildfire burn-scar segmentation under the tested linear, nonlinear pointwise, and
> lightweight spatial readouts. Conventional spectral representations remained
> substantially more transferable under the same event-disjoint evaluation framework.**

We deliberately distinguish — and do **not** conflate — five separate claims:

| Claim | Verdict in this study |
|---|---|
| linear transferability of frozen features | weak |
| nonlinear readout rescues transfer | no (improves ranking, not segmentation) |
| spatial decoding rescues transfer | no |
| frozen-representation transfer overall | poor |
| encoder adaptation / task-specific fine-tuning needed | **not tested** — a limitation, not a finding |

We do **not** claim that Prithvi is generally a poor foundation model, that foundation models
cannot work for wildfire mapping, that the frozen features contain *zero* burn information, that
decoder capacity has been universally ruled out, or that the entire gap is *proven* to arise from
representation content alone.

---

## 2. Why event-disjoint evaluation matters

Random train/test splits over-predict real-world wildfire mapping performance because training and
test pixels share spatial context (the same scene, tile, and local environment) and sensor/date
context. A model that memorizes scene appearance scores well but does not transfer to a genuinely
new fire. This study therefore evaluates exclusively on **unseen events**: a model is trained on a
subset of fire events and scored only on events it never saw, using a **shared, deterministic,
event-disjoint 5-fold cross-validation**. Pixel-level random splitting is forbidden throughout;
pixels are never treated as independent statistical units.

---

## 3. Dataset and MTBS event reconstruction

- **Source:** `ibm-nasa-geospatial/hls_burn_scars` — 512×512 HLS chips, 6 bands (B02, B03, B04,
  B8A, B11, B12), CONUS 2018–2021, labels derived from MTBS burn-scar perimeters.
- **Event identity:** the dataset has no explicit `event_id`. Events were reconstructed by matching
  chip geometry (CRS, affine transform, bounds, acquisition date) to MTBS fire perimeters via
  spatial intersection + temporal consistency + burn-mask overlap.
- **Primary population:** 576 reliably matched events with MTBS `Incid_Type == "Wildfire"` (one
  post-fire chip per event in this dataset). Prescribed Fire (177), Unknown, ambiguous, and
  unmatched events are excluded from the primary benchmark.
- **Reconstruction products:** `data/metadata/events.csv`, `event_attributes.csv`,
  `chip_inventory.csv`, plus split manifests.

---

## 4. Experimental protocol

- **Split:** deterministic 5-fold event-disjoint CV, seed 42, stratified by (fire_year × quantiles
  of burned fraction), **one shared fold assignment for all models**
  (`data/metadata/splits/split_k5_event_inner_seed42.csv`).
- **Supervision budget:** ≤ 2048 sampled valid training pixels per chip (identical across models),
  sampled with a per-chip seeded RNG.
- **Loss / optimizer:** `BCEWithLogitsLoss` with per-fold `pos_weight` from inner-train class ratio
  only; AdamW, lr `1e-3`, weight decay `1e-4`, max 50 epochs, patience 5 on inner-val burned Dice.
  No Dice/focal loss.
- **Threshold:** selected once per fold on the frozen inner-val pool (99-point grid, maximize pooled
  burned Dice); the outer test set is touched once per fold.
- **Statistics:** event-level metrics (IoU, Dice, precision, recall, AUPRC, AUROC), summarized with
  mean / median and percentile-bootstrap 95% CIs (10,000 resamples, seed 42); paired per-event
  deltas with the same bootstrap.

---

## 5. Spectral baseline

8 spectral features (6 HLS bands + NDVI + NBR) → Random Forest, pixel-wise segmentation.

- Event-level OOF: **IoU 0.564 / Dice 0.682 / AUROC 0.930** (mean), AUPRC 0.770.
- The RF generalizes across the full burn-prevalence range and localizes the scar: its
  boundary-to-interior error profile falls from ~0.41 (edge) to ~0.03 (deep interior).

## 6. Frozen Prithvi linear-transfer result

Frozen Prithvi-EO-2.0-300M encoder, layers `[5, 11, 17, 23]` concatenated to a 4096-d per-token
feature (32×32 grid), read out by a single 1×1 conv + bilinear upsampling (4,097 trainable
parameters).

- Event-level OOF: **IoU 0.131 / Dice 0.197 / AUROC 0.578** (mean), AUPRC 0.159.
- The linear probe over-predicts (recall 0.871, precision 0.142; pred/true burn-fraction correlation
  0.009) and is wrong *everywhere*, not only at boundaries.

## 7. Nonlinear and spatial readout controls

Two controls test whether the linear probe's failure was a *readout* problem rather than a
*representation* problem.

- **Pointwise MLP** (1,065,345 params; 1×1 convs only, no spatial context): **IoU 0.129 / Dice
  0.218 / AUROC 0.652**. Nonlinearity recovers ranking signal the linear probe missed (AUROC +0.074,
  AUPRC +0.078, both significant) and fixes the over-prediction pathology — but **does not improve
  segmentation** (Δ IoU vs linear −0.001, n.s.).
- **Lightweight spatial decoder** (655,553 params; per-layer 1×1 projections + 3×3 convs +
  progressive bilinear upsampling): **IoU 0.155 / Dice 0.233 / AUROC 0.578**. Spatial context adds
  only +0.024 IoU over the linear probe and actually *lowers* AUROC (0.652 → 0.578). Its error
  profile is flat-to-rising — wrong everywhere, not just at edges.

## 8. Matched spectral spatial control

The decisive control: the **same** spatial-decoder architecture and **same** training protocol, fed
the **8 spectral channels** instead of frozen Prithvi features (523,809 params — within ~25 % of the
Prithvi decoder).

- Event-level OOF: **IoU 0.577 / Dice 0.679 / AUROC 0.935**, AUPRC 0.796.
- The spectral CNN is statistically indistinguishable from the RF (IoU 0.577 vs 0.564; overlapping
  CIs) and tracks it across every burn-fraction bin.

Because the *only* thing that changes between the Prithvi decoder and this control is the **input
representation**, the resulting gap isolates representation content from decoder capacity.

---

## 9. Main findings

### 9.1 Canonical five-model table (event-level OOF, 576 events, mean / median)

| Model | Input representation | Trainable head | Spatial context | Encoder frozen? | Trainable params | IoU | Dice | Precision | Recall | AUPRC | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Spectral RF | 8 spectral feat. | Random Forest | none | — | — | 0.564 / 0.616 | 0.682 / 0.763 | 0.775 | 0.688 | 0.770 / 0.853 | 0.930 / 0.953 |
| Spectral CNN | 8 spectral chan. | spatial CNN | 3×3 conv, down/up | — | 523,809 | 0.577 / 0.666 | 0.679 / 0.799 | 0.751 | 0.727 | 0.796 / 0.929 | 0.935 / 0.990 |
| Prithvi linear | frozen Prithvi 4096-d | 1×1 conv + bilinear | none | **yes** | 4,097 | 0.131 / 0.053 | 0.197 / 0.101 | 0.142 | 0.871 | 0.159 / 0.078 | 0.578 / 0.571 |
| Prithvi pointwise MLP | frozen Prithvi 4096-d | pointwise MLP | none | **yes** | 1,065,345 | 0.129 / 0.110 | 0.218 / 0.198 | 0.206 | 0.452 | 0.238 / 0.185 | 0.652 / 0.636 |
| Prithvi spatial decoder | frozen Prithvi 4×1024-d | light spatial decoder | 3×3 conv, up | **yes** | 655,553 | 0.155 / 0.072 | 0.233 / 0.134 | 0.205 | 0.850 | 0.209 / 0.101 | 0.578 / 0.524 |

<sup>†</sup> Spectral-CNN precision is defined on 556/576 events (20 events had zero predicted
positives); this does not affect IoU/Dice/AUROC, which use all 576.

### 9.2 Bootstrap 95% CIs (event-level, 10,000 resamples, seed 42)

| Metric | Spectral CNN | Spectral RF | Prithvi decoder | Prithvi MLP | Prithvi linear |
|---|---|---|---|---|---|
| IoU | 0.577 [0.554, 0.601] | 0.564 [0.543, 0.584] | 0.155 [0.140, 0.170] | 0.129 [0.122, 0.137] | 0.131 [0.116, 0.146] |
| Dice | 0.679 [0.654, 0.703] | 0.682 [0.662, 0.702] | 0.233 [0.215, 0.252] | 0.218 [0.207, 0.229] | 0.197 [0.180, 0.215] |
| AUROC | 0.935 [0.922, 0.946] | 0.930 [0.924, 0.936] | 0.578 [0.567, 0.589] | 0.652 [0.643, 0.660] | 0.578 [0.573, 0.583] |

### 9.3 Paired per-event deltas (paired bootstrap 95% CIs)

The decisive comparison — matched spatial capacity, differing only in input representation:

| Δ (model_a − model_b) | IoU | Dice | AUPRC | AUROC |
|---|---|---|---|---|
| **decoder − spectral CNN** | **−0.422 [−0.447, −0.397]** | **−0.446 [−0.472, −0.418]** | **−0.587 [−0.612, −0.563]** | **−0.357 [−0.373, −0.341]** |
| decoder − spectral RF | −0.409 [−0.432, −0.386] | −0.449 [−0.473, −0.425] | −0.561 [−0.583, −0.538] | −0.352 [−0.365, −0.339] |
| MLP − spectral RF | −0.435 [−0.454, −0.415] | −0.465 [−0.483, −0.445] | −0.532 [−0.551, −0.513] | −0.278 [−0.288, −0.269] |

Readout-capacity controls (within the Prithvi family):

| Δ | IoU | Dice | AUPRC | AUROC |
|---|---|---|---|---|
| MLP − linear | −0.001 [−0.012, +0.009] (n.s.) | +0.021 [+0.010, +0.032] | +0.078 [+0.072, +0.085] | +0.074 [+0.066, +0.081] |
| decoder − linear | +0.024 [+0.016, +0.033] | +0.036 [+0.025, +0.048] | +0.050 [+0.038, +0.062] | −0.000 [−0.012, +0.012] (n.s.) |

### 9.4 Sequence of evidence

1. **RF strong** (IoU 0.564).
2. **Spatial leakage ruled out as the major explanation** — the primary folds are event-disjoint,
   and the matched-capacity spectral control confirms the pipeline is sound.
3. **Prithvi linear fails** (IoU 0.131).
4. **Nonlinear probe improves ranking but not segmentation** (AUROC 0.578→0.652; IoU flat).
5. **Spatial decoder fails to recover transfer** (IoU 0.155; AUROC falls back to 0.578).
6. **Spectral CNN with comparable spatial modeling remains strong** (IoU 0.577).
7. **The frozen Prithvi representation pipeline is the primary observed limitation.**

---

## 10. Failure analysis

- **Low burned fraction is the dominant failure regime.** The abundant small-burn events (0.01–0.05
  burned fraction, n=283) show mean IoU ~0.49 for both spectral models versus 0.03–0.08 for the
  Prithvi readouts. The Prithvi probes only become non-trivial on near-solid, high-fraction scars.
- **The Prithvi probes fail by mis-localization, not mere under-confidence.** Their
  boundary-to-interior error profiles are flat-to-rising (wrong everywhere); the spectral models
  localize the scar and err mainly at its edge.
- **Calibration collapse.** The Prithvi linear probe and decoder collapse toward the base rate
  (pred/true burn-fraction correlation 0.009–0.016), whereas the spectral models are nearly unbiased
  (r ≈ 0.91).
- **A rare Prithvi-favorable case exists.** On the largest, most obvious scar in the population
  (burned fraction 0.936), the decoder reaches IoU 0.931 where the spectral CNN reaches 0.708. The
  Prithvi probes succeed only where the burn is a near-solid scar.

## 11. Limitations

1. **Event ≈ one scene.** In `hls_burn_scars` each reliably matched wildfire event is a single
   512×512 chip, so "event" and "scene" coincide and multi-scene events are not represented.
2. **Single-date imagery.** Only post-fire scenes are used; there is no pre/post temporal contrast
   and no dNBR.
3. **Foundation-model pretraining exposure.** "Unseen wildfire event" means unseen during
   *downstream* training; we do **not** establish that the geography or imagery was unseen during
   foundation-model pretraining.
4. **Frozen-only evaluation.** Results do not establish what happens after encoder fine-tuning.
5. **One foundation-model family / checkpoint.** Only Prithvi-EO-2.0-300M (foundation checkpoint)
   was tested.
6. **Tokenization / representation pipeline differ.** Prithvi's tokenization and normalization
   differ from the raw spectral inputs, so the comparison does not isolate every possible mechanism
   of information loss.
7. **Decoder search deliberately limited.** Failure of the tested decoder does not prove that no
   possible decoder could recover useful performance.
8. **Dataset scope.** Primarily CONUS 2018–2021 MTBS-linked events.
9. **Prescribed fires excluded** from the primary benchmark.
10. **Small burned fractions** are a major failure regime for all models.

## 12. Future work (not executed — separate from current findings)

- **A.** Partial or full Prithvi fine-tuning, to test whether task-specific adaptation restores
  transfer.
- **B.** Compare against a **BurnScars-finetuned Prithvi as a positive control**, clearly separated
  from the frozen-representation experiment.
- **C.** Multi-temporal pre/post-fire imagery and dNBR.
- **D.** Other EO foundation models.
- **E.** Explicit task-specific contrastive / segmentation adaptation.
- **F.** Prescribed-fire domain-shift evaluation.

## 13. Reproducibility

- Environment pinned in `environment.yml` / `requirements-lock.txt`; data and cache paths are
  env-driven (`PRITHVI_WF_ROOT`), large data/weights/caches gitignored.
- Seeds recorded: split seed 42, bootstrap seed 42, per-chip sampling via `chip_rng(SEED, image_id)`.
  Foundation checkpoint `Prithvi_EO_V2_300M.pt` (frozen, 303,886,336 params).
- Exact per-stage commands and inputs/outputs are in `docs/REPRODUCIBILITY.md`.
- Result CSVs that feed the figures are committed under `results/m5_compare/`; per-event prediction
  maps (`*.npy`) are regenerable and not committed.

---

### Outputs

- Canonical table + deltas: `results/m5_compare/` (model matrix, ranking, paired deltas,
  burn-fraction, calibration, hardest/easiest, boundary error).
- Final figures: `results/figures/final/fig{1..6}_*.png` (generated by `scripts/m6_make_figures.py`).
- Abstract: `docs/ABSTRACT.md`; one-page summary: `docs/ONE_PAGE_RESEARCH_SUMMARY.md`.

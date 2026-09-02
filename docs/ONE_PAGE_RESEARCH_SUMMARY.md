# One-Page Research Summary

## Question

Do **frozen Prithvi-EO-2.0-300M representations** transfer better than **conventional
spectral features** for burned-area segmentation on **entirely unseen wildfire events**?

**Short answer: no, not under the readouts we tested.** Frozen Prithvi transferred poorly
(event IoU ≈ 0.13–0.16); conventional spectral features remained far more transferable
(event IoU ≈ 0.56–0.58) under an identical event-disjoint evaluation.

## Method

- **576 MTBS-linked wildfire events** (CONUS 2018–2021, one post-fire HLS scene each),
  reconstructed by matching chip geometry to MTBS fire perimeters.
- **Shared, deterministic event-disjoint 5-fold CV** (seed 42) — every model trained and
  tested on the *same* folds; no pixel-level splitting; one out-of-fold prediction per event.
- **Five models**, all under an identical supervision protocol (≤2048 px/chip, BCE, AdamW,
  inner-val Dice threshold):

  | # | Model | Input | Trainable params |
  |---|---|---|---|
  | 1 | Spectral RF | 8 spectral features (6 bands + NDVI + NBR) | — |
  | 2 | Spectral CNN | 8 spectral channels | 523,809 |
  | 3 | Prithvi linear | frozen 4096-d features | 4,097 |
  | 4 | Prithvi pointwise MLP | frozen 4096-d features | 1,065,345 |
  | 5 | Prithvi spatial decoder | frozen 4×1024-d features | 655,553 |

- **Statistics:** event-level metrics, bootstrap 95% CIs (10,000 resamples), paired deltas.

## Key result

| Model | IoU | Dice | AUROC |
|---|---|---|---|
| Spectral RF | 0.564 | 0.682 | 0.930 |
| Spectral CNN | 0.577 | 0.679 | 0.935 |
| Prithvi decoder | 0.155 | 0.233 | 0.578 |
| Prithvi MLP | 0.129 | 0.218 | 0.652 |
| Prithvi linear | 0.131 | 0.197 | 0.578 |

The decisive control feeds the **same spatial decoder** either frozen Prithvi features or
the 8 spectral channels. **Decoder − spectral CNN = −0.422 IoU [−0.447, −0.397].** Because
only the input representation changes, the gap is attributable to **representation
content**, not decoder capacity. Nonlinearity (MLP) raises AUROC (0.578→0.652) but not IoU;
spatial decoding does not help either.

## Why it matters

Cross-event generalization is what wildfire mapping actually needs in deployment — new
fires, new geographies. This study shows that a widely used EO foundation model's *frozen*
representations do not automatically transfer to burn-scar segmentation under a clean
event-disjoint protocol, and that a cheap spectral baseline remains strong. It provides a
reproducible, negative-but-informative result: the bottleneck is not the readout but the
frozen representation's weak transferable burn signal, pointing toward encoder adaptation
as the next lever.

## Limitations

- Event ≈ one scene (single 512×512 chip); single-date imagery (no dNBR).
- "Unseen event" = unseen in downstream training; pretraining exposure not controlled.
- Frozen-only evaluation — fine-tuning not tested; only one foundation checkpoint.
- Decoder search deliberately limited — failure of tested decoders is not proof that no
  decoder could succeed. Small burned fractions are the dominant failure regime.

## Next experiment

**Fine-tune (or partially adapt) the Prithvi encoder** on the burn-scar task and compare
against a **BurnScars-finetuned positive control**, still under the event-disjoint
protocol. This directly tests whether task-specific adaptation — rather than the frozen
representation — restores cross-event transfer.

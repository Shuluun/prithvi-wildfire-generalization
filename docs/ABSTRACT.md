# Abstract

Cross-event generalization is the deployment-critical question for wildfire burn-scar
mapping, yet random train/test splits can overstate performance because training and test
samples share spatial, environmental, and sensor context. We ask whether **frozen**
representations from the Prithvi-EO-2.0-300M foundation model transfer better than
conventional spectral features for burned-area segmentation on entirely unseen wildfire
events. Using 576 MTBS-linked wildfire events (CONUS 2018–2021, one post-fire HLS scene
each), we enforce a shared event-disjoint 5-fold cross-validation and compare a spectral
Random Forest baseline, a matched-capacity spectral CNN control, and three frozen-Prithvi
readouts — a linear probe, a nonlinear pointwise MLP, and a lightweight spatial decoder.
The spectral models generalize strongly (event-level mean IoU 0.564–0.577, AUROC ≈ 0.93),
whereas the frozen-Prithvi readouts reach only IoU 0.13–0.16. Nonlinearity recovers signal
the linear probe missed (AUROC 0.58 → 0.65) but does not improve segmentation; spatial
decoding likewise fails to close the gap. Because the matched spectral CNN — the same
spatial decoder fed 8 spectral channels instead of frozen features — recovers
near-baseline performance, the gap is attributable to the frozen representation's weak
transferable burn signal rather than to readout capacity. We interpret this as evidence
that frozen foundation-model representations do not transfer automatically to cross-event
burn-scar segmentation and that encoder adaptation or task-specific fine-tuning may be
required. We do not conclude that no decoder, no foundation model, or no fine-tuning
regime could succeed.

**Keywords:** wildfire, burn scar, semantic segmentation, foundation model, Prithvi-EO-2.0,
cross-event generalization, event-disjoint evaluation.

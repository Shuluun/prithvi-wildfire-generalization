# Data

This repository tracks the **small metadata** needed to audit and reproduce the study's event
linkage and splits. The **raw** HLS/MTBS imagery, model weights, and the frozen feature cache are
large and are stored outside the repository (see `src/data/paths.py` for the `PRITHVI_WF_ROOT`
storage-root convention).

## `metadata/`

- `chip_inventory.csv` — one row per HLS chip (image id, HLS tile, split, acquisition info).
- `events.csv` — reconstructed event identity (chip ↔ MTBS fire-perimeter match).
- `event_attributes.csv` — per-event attributes for the primary population (incident type, fire
  year, burned fraction, HLS tile).
- `splits/` — deterministic split manifests (seed 42). The primary protocol is the shared 5-fold
  event-disjoint split (`split_k5_event_*.csv`) plus its spatial-overlap and HLS-tile-disjoint
  sensitivity variants. `split_loeo_seed42.csv` is regenerable and not committed.

"""M1.5: write deterministic split manifests (conventional + LOEO)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT  # noqa: E402
from src.data.splits import write_manifests  # noqa: E402

if __name__ == "__main__":
    inv = pd.read_csv(os.path.join(METADATA_ROOT, "chip_inventory.csv"))
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    chip_table = inv[["image_id"]].merge(
        ev[["image_id", "event_id", "match_status"]], on="image_id",
        how="left")
    within, loeo = write_manifests(chip_table, seed=42)
    print("conventional split:")
    print(within["split"].value_counts().to_string())
    print(f"\nLOEO manifest: {len(loeo)} rows, "
          f"{loeo['held_out_event'].nunique()} folds, "
          f"{len(loeo[loeo['role']=='test'])} test rows")

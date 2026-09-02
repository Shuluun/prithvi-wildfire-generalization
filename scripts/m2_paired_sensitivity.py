"""M2 paired PRIMARY-vs-SPATIAL sensitivity (event-level, paired bootstrap).

For every primary Wildfire event, merges the PRIMARY (event) and SPATIAL
out-of-fold predictions by event_id and computes the per-event delta:

    delta = spatial - primary

for the four ranking/overlap metrics, then summarizes each delta with mean,
median, and a percentile bootstrap 95% CI over events. This is the proper
quantitative assessment of enforcing zero direct train/test footprint overlap:
the two marginal CIs are NOT overlapped as a significance argument.

Usage (from repo root):
  python scripts/m2_paired_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import RESULTS_ROOT  # noqa: E402
from src.evaluation.bootstrap import bootstrap_ci  # noqa: E402

DELTA_METRICS = ["iou", "dice", "auprc", "auroc"]


def main():
    event_dir = os.path.join(RESULTS_ROOT, "m2", "event")
    spatial_dir = os.path.join(RESULTS_ROOT, "m2", "spatial")

    ev = pd.read_csv(os.path.join(event_dir, "m2_event_oof_predictions.csv"))
    sp = pd.read_csv(os.path.join(spatial_dir, "m2_event_oof_predictions.csv"))

    assert ev["event_id"].is_unique and sp["event_id"].is_unique
    assert set(ev["event_id"]) == set(sp["event_id"])

    merged = ev.merge(sp, on="event_id", suffixes=("_primary", "_spatial"))
    assert len(merged) == len(ev) == len(sp)

    rows = []
    print("=== paired PRIMARY-vs-SPATIAL delta (spatial - primary) ===")
    print(f"events merged: {len(merged)}\n")
    for m in DELTA_METRICS:
        delta = (merged[f"{m}_spatial"] - merged[f"{m}_primary"]).to_numpy(
            dtype=float)
        finite = delta[np.isfinite(delta)]
        n_dropped = int(len(delta) - len(finite))
        lo, hi = bootstrap_ci(delta, seed=42)
        frac_spatial_higher = float((delta > 0).mean()) if len(finite) else np.nan
        rows.append({
            "metric": m,
            "n_events": int(len(finite)),
            "n_nan_pairs_dropped": n_dropped,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "median": float(np.median(finite)) if len(finite) else np.nan,
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
            "frac_spatial_higher": frac_spatial_higher,
        })
        print(f"{m:8s} mean={finite.mean():+.4f} median={np.median(finite):+.4f} "
              f"CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"(spatial>primary in {frac_spatial_higher:.1%} of events, "
              f"{n_dropped} NaN pairs dropped)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_ROOT, "m2", "m2_paired_sensitivity.csv")
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    # persist the merged per-event deltas for the figures/report
    delta_cols = [f"delta_{m}" for m in DELTA_METRICS]
    merged_delta = pd.DataFrame({
        "event_id": merged["event_id"],
        "image_id": merged["image_id_primary"],
    })
    for m in DELTA_METRICS:
        merged_delta[f"delta_{m}"] = (
            merged[f"{m}_spatial"] - merged[f"{m}_primary"])
    merged_delta.to_csv(
        os.path.join(RESULTS_ROOT, "m2", "m2_paired_deltas_per_event.csv"),
        index=False)
    print(f"-> {os.path.join(RESULTS_ROOT, 'm2', 'm2_paired_deltas_per_event.csv')}")
    return out


if __name__ == "__main__":
    main()

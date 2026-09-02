"""M1 CLI: reconstruct fire-event linkage for hls_burn_scars chips.

Usage (run from repo root):
  python scripts/m1_reconstruct_events.py --pilot 100
  python scripts/m1_reconstruct_events.py --full

Pilot mode: run matching on the first N chips of training/ and write
candidate scores + a distribution summary (method validation).
Full mode: run all chips, classify with the thresholds (finalized from the
full 790-candidate distribution — the `--pilot 804` candidate stage,
matching_summary_pilot804.txt; see the M1 report §4 and the M1.6
reconciliation), write data/metadata/events.csv.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.event_reconstruction import (  # noqa: E402
    CANONICAL_COLUMNS, classify, iter_chips, load_perimeters, run_matching,
)
from src.data.paths import (  # noqa: E402
    HLS_BURN_SCARS_DIR, METADATA_ROOT, MTBS_SHP_DIR, RESULTS_ROOT,
)


def summary(df, out_dir, tag):
    """Write distribution summary for threshold selection."""
    lines = [f"# matching summary ({tag})", f"n_chips = {len(df)}"]
    cand = df[df["match_status"] == "candidate"]
    lines.append(f"n_candidate_rows = {len(cand)}")
    lines.append(f"n_unmatched (no candidate) = {(df['match_status'] == 'unmatched').sum()}")
    if len(cand):
        for col in ("match_confidence", "intersection_ratio",
                    "burn_mask_overlap", "temporal_gap_days",
                    "second_score_margin"):
            s = cand[col]
            lines.append(
                f"{col}: min={s.min():.4f} q25={s.quantile(.25):.4f} "
                f"median={s.median():.4f} q75={s.quantile(.75):.4f} "
                f"max={s.max():.4f}")
    txt = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, f"matching_summary_{tag}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, metavar="N",
                   help="run matching on the first N chips of each split dir")
    g.add_argument("--full", action="store_true", help="run all chips")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="classification threshold (from pilot distribution)")
    ap.add_argument("--min-inter-ratio", type=float, default=None)
    args = ap.parse_args()

    gdf, tree = load_perimeters(MTBS_SHP_DIR)
    print(f"perimeters: {len(gdf)} rows, crs={gdf.crs}")

    rows = []
    tag = f"pilot{args.pilot}" if args.pilot else "full"
    for split_dir in ("training", "validation"):
        pairs = list(iter_chips(HLS_BURN_SCARS_DIR, split_dir))
        if args.pilot:
            pairs = pairs[:args.pilot]
        print(f"{split_dir}: matching {len(pairs)} chips ...")
        df = run_matching(pairs, gdf, tree)
        rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    out_dir = os.path.join(RESULTS_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"matching_{tag}.csv"), index=False)

    if args.full:
        if args.min_confidence is None or args.min_inter_ratio is None:
            print("--full requires --min-confidence and --min-inter-ratio "
                  "(chosen from the pilot summary)")
            sys.exit(2)
        df = classify(df, args.min_confidence, args.min_inter_ratio)
        os.makedirs(METADATA_ROOT, exist_ok=True)
        events_csv = os.path.join(METADATA_ROOT, "events.csv")
        df[CANONICAL_COLUMNS].to_csv(events_csv, index=False)
        print(f"events.csv written: {events_csv}")
        print(df["match_status"].value_counts().to_string())

    summary(df, out_dir, tag)


if __name__ == "__main__":
    main()

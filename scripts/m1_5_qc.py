"""M1.5: benchmark QC — distributions + figures from events.csv / inventory.

Produces (results/reports/ and results/figures/):
  qc_summary.txt                       — all distribution numbers
  fig_qc_acq_dates.png                 — acquisition dates by month
  fig_qc_chip_locations.png            — chip centroids by match status
  fig_qc_overlap_neighbors.png         — per-event geographic neighbor count
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.paths import METADATA_ROOT, RESULTS_ROOT  # noqa: E402

REPORTS = os.path.join(RESULTS_ROOT, "reports")
FIGURES = os.path.join(RESULTS_ROOT, "figures")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    ev = pd.read_csv(os.path.join(METADATA_ROOT, "events.csv"))
    inv = pd.read_csv(os.path.join(METADATA_ROOT, "chip_inventory.csv"))
    m = ev.merge(inv[["image_id", "split", "burned_pixels", "unburned_pixels",
                      "missing_pixels", "crs_wkt"]], on="image_id")

    L = []
    L.append("=== M1.5 QC distributions ===")

    # 1. event sample-count distribution
    matched = m[m["match_status"] == "matched"]
    n_events = matched["event_id"].nunique()
    L.append(f"events: {n_events} unique; chips/event: all exactly 1 "
             f"(structural property, see M1 report)")

    # 2. class balance (per-chip burned share)
    share = m["burned_pixels"] / (m["burned_pixels"] + m["unburned_pixels"])
    L.append("per-chip burned-pixel share: "
             f"min={share.min():.4f} q25={share.quantile(.25):.4f} "
             f"median={share.median():.4f} q75={share.quantile(.75):.4f} "
             f"max={share.max():.4f}")
    tot = m[["burned_pixels", "unburned_pixels", "missing_pixels"]].sum()
    L.append(f"pooled pixels: {tot.to_dict()}; burned share pooled="
             f"{tot['burned_pixels']/(tot['burned_pixels']+tot['unburned_pixels']):.4f}")

    # 3. acquisition-date distribution
    m["acq"] = pd.to_datetime(m["acquisition_date"])
    L.append("acquisition dates: "
             f"{m['acq'].min().date()} .. {m['acq'].max().date()}")
    yr = m["acq"].dt.year.value_counts().sort_index()
    L.append("chips per year:\n" + yr.to_string())
    plt.figure(figsize=(7, 3))
    m["acq"].dt.to_period("M").value_counts().sort_index().plot.bar(width=.9)
    plt.title("chip acquisition dates by month")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_qc_acq_dates.png"), dpi=90)
    plt.close()

    # 4. spatial extent: chip centroids (WGS84 bounds from events.csv)
    def centroid(r):
        g = wkt.loads(r)
        return g.centroid.x, g.centroid.y
    m["lon"], m["lat"] = zip(*m["chip_bounds"].apply(centroid))
    colors = {"matched": "#1a9850", "ambiguous": "#fdae61", "unmatched": "#d73027"}
    plt.figure(figsize=(9, 5))
    for st, grp in m.groupby("match_status"):
        plt.scatter(grp["lon"], grp["lat"], s=4, c=colors[st], label=st)
    plt.legend(markerscale=3)
    plt.title(f"chip centroids by match status (n={len(m)})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_qc_chip_locations.png"), dpi=90)
    plt.close()

    # 5. geographic overlap: per-event neighbor count
    ov = pd.read_csv(os.path.join(REPORTS, "event_overlap_pairs.csv"))
    L.append(f"overlapping event pairs: {len(ov)}")
    if len(ov):
        L.append(f"pair overlap area (km^2): min={ov['overlap_area_m2'].min()/1e6:.2f} "
                 f"median={ov['overlap_area_m2'].median()/1e6:.2f} "
                 f"max={ov['overlap_area_m2'].max()/1e6:.2f}")
        neigh = pd.concat([ov["event_a"], ov["event_b"]]).value_counts()
        L.append(f"events with >=1 overlapping neighbor: {len(neigh)} "
                 f"of {n_events} ({len(neigh)/n_events:.1%})")
        L.append(f"neighbor count: max={neigh.max()}, "
                 f"median={neigh.median():.1f}")
        plt.figure(figsize=(6, 3))
        neigh.value_counts().sort_index().plot.bar(width=.9)
        plt.title("geographic neighbor count per event")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES, "fig_qc_overlap_neighbors.png"), dpi=90)
        plt.close()

    # 6. ambiguous / unmatched diagnostics
    amb = m[m["match_status"] == "ambiguous"]
    unm = m[m["match_status"] == "unmatched"]
    L.append(f"ambiguous: {len(amb)} | unmatched: {len(unm)} "
             f"| excluded share: {(len(amb)+len(unm))/len(m):.1%}")
    L.append(f"matched per split: {matched['split'].value_counts().to_dict()}")
    L.append("fire_year counts (matched):\n" +
             matched["fire_year"].value_counts().sort_index().to_string())

    txt = "\n".join(L) + "\n"
    with open(os.path.join(REPORTS, "qc_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()

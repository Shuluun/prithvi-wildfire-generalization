"""Event-level bootstrap inference.

Pixels are NOT independent statistical units: every statistic is computed per
event (= per chip in hls_burn_scars), then summarized over events with a
percentile bootstrap over 10,000 resamples of the event-level values (and, for
paired deltas, of the per-event deltas).
"""
import numpy as np
import pandas as pd


def bootstrap_ci(values, n_resamples=10000, seed=42, alpha=0.05):
    """Percentile bootstrap 95% CI of the MEAN over events.

    values: 1-D array of event-level values (NaNs dropped). Returns
    (lo, hi).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_events(df, metric_columns, n_resamples=10000, seed=42):
    """Event-level summary table: mean/median/std/quantiles/bootstrap CI.

    df: one row per event with the metric columns present. Returns a DataFrame
    with one row per metric.
    """
    rows = []
    for col in metric_columns:
        vals = df[col].astype(float)
        finite = vals[np.isfinite(vals)]
        lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        rows.append({
            "metric": col,
            "n_events": int(len(finite)),
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "median": float(finite.median()) if len(finite) else float("nan"),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "q05": float(np.nanquantile(vals, 0.05)),
            "q25": float(np.nanquantile(vals, 0.25)),
            "q75": float(np.nanquantile(vals, 0.75)),
            "q95": float(np.nanquantile(vals, 0.95)),
            "bootstrap_ci_lo": lo,
            "bootstrap_ci_hi": hi,
        })
    return pd.DataFrame(rows)

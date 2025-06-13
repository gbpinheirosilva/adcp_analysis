"""
adcp_stats.py
=============

Analysis helpers that work on *speed_mat* (rows = BINs, cols = timestamps)
returned by `adcp_grids.build_metric_grids`.

Key functions
-------------
compare_bins(speed_mat, bins, start, end)
validity_report(speed_mat, bins, start, end)
histogram_validity(speed_mat, start, end [, ax])
"""
from __future__ import annotations
from typing import List, Tuple, Sequence

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------
def _slice_window(
    df: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return df restricted to time-window on the column axis."""
    sliced = df
    if start is not None:
        sliced = sliced.loc[:, sliced.columns >= pd.to_datetime(start)]
    if end is not None:
        sliced = sliced.loc[:, sliced.columns <= pd.to_datetime(end)]
    return sliced


# ------------------------------------------------------------------
# ─── adcp_stats.py (replace the old compare_bins) ─────────────────
def compare_bins(
    speed_mat: pd.DataFrame,
    bins: Sequence[str] = ("BIN0", "BIN1", "BIN2"),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    markersize=4,
    linewidth=1,
    colors=None,     
):
    """
    Plot up to 3 bins against each other over time.
    Drops bins that are non-numeric *or* entirely NaN in the chosen window,
    and prints a friendly warning.
    """
    df = _slice_window(speed_mat, start, end).reindex(index=bins).T

    # NEW ▶︎  force numeric & keep track of bins that vanish
    before_cols = df.columns.tolist()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")           # drop all-NaN bins
    gone = [b for b in before_cols if b not in df.columns]

    if gone:
        print(f"⚠️  Skipped non-numeric or all-NaN bins: {', '.join(gone)}")

    if df.empty:
        raise ValueError("No numeric data to plot in the selected window.")

    df.plot(figsize=(10, 4), marker="o", markersize=markersize,linewidth=linewidth,color=colors)
    plt.title(f"Speed comparison: {', '.join(df.columns)}")
    plt.xlabel("Timestamp")
    plt.ylabel("Speed (m/s)")
    plt.grid(True)
    plt.tight_layout()
# ──────────────────────────────────────────────────────────────────



# ------------------------------------------------------------------
def validity_report(
    speed_mat: pd.DataFrame,
    bins: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    expected_interval: pd.Timedelta = pd.Timedelta(minutes=20),
) -> pd.DataFrame:
    """
    Return a DataFrame with invalid_cnt, total_cnt, valid_pct for the bins.

    expected_interval is *only* used to compute the theoretical expected
    number of measurements (total_cnt) if you prefer that; otherwise the
    function uses the real column count.
    """
    df = _slice_window(speed_mat, start, end)
    if bins is not None:
        df = df.reindex(index=bins)

    timeline_len = df.shape[1]

    stats = {
        bin_: {
            "invalid_cnt": int(row.isna().sum()),
            "total_cnt":   timeline_len,
            "valid_pct":   100 * (1 - row.isna().sum() / timeline_len)
        }
        for bin_, row in df.iterrows()
    }

    return pd.DataFrame(stats).T  # rows=bins


# ------------------------------------------------------------------
def histogram_validity(
    speed_mat: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    ax=None,
):
    """
    Plot %-valid per BIN (all rows) for the requested time-window.
    """
    stats = validity_report(speed_mat, start=start, end=end)
    percentages = stats["valid_pct"]

    if ax is None:
        ax = plt.gca()

    percentages.plot.bar(ax=ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("BIN")
    ax.set_ylabel("Valid data (%)")
    ax.set_title("Validity by BIN")
    ax.grid(True, axis="y")

    plt.tight_layout()


# ─── NEW: quick validity counters & daily histogram ───────────────
def valid_bin_counts(
    speed_mat: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Return a Series indexed by *timestamp* whose value is the number of
    bins (rows) that are NON-null at that moment.
    """
    window = _slice_window(speed_mat, start, end)
    return window.notna().sum(axis=0)      # counts across rows


def plot_daily_valid_hist(
    speed_mat: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    agg: str = "mean",                     # "mean", "max", or "sum"
):
    """
    Histogram: X-axis = day, Y-axis = aggregated valid-bin count.

    • 'mean' → average #valid bins per ensemble that day  
    • 'max'  → best ensemble that day  
    • 'sum'  → total valid samples  (careful: depends on ensemble rate)
    """
    counts = valid_bin_counts(speed_mat, start, end)
    daily  = counts.groupby(counts.index.normalize())

    if   agg == "mean":
        daily_counts = daily.mean()
        ylabel = "Mean valid bins / ensemble"
    elif agg == "max":
        daily_counts = daily.max()
        ylabel = "Max valid bins in any ensemble"
    elif agg == "sum":
        daily_counts = daily.sum()
        ylabel = "Total valid samples"
    else:
        raise ValueError("agg must be 'mean', 'max', or 'sum'")

    ax = daily_counts.plot.bar(figsize=(12,4))
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    ax.set_title(f"Daily {agg} of valid-bin count")
    ax.set_ylim(0, speed_mat.shape[0])     # 0-to-max-bins scale
    ax.grid(True, axis="y")
    plt.tight_layout()

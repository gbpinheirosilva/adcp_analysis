"""
adcp_grids.py
=============

Builds two "matrix" DataFrames (speed, direction) whose

* **rows** .... BIN0, BIN1, … BIN29      (or any range you pass)
* **columns** . measurement timestamps   (taken from the CS records)

BIN0 comes from the $PNORCA sentences, which are not timestamp-aligned
with $PNORCS.  We therefore snap each CA record to the *closest* CS
timestamp (earlier or later, whichever has the smaller Δt) – this mapping
is computed once and reused.

Typical use
-----------
>>> from adcp_parser import parse_file
>>> from adcp_grids  import build_metric_grids
>>>
>>> ca_df, cs_df = parse_file("raw.txt")
>>> speed_df, dir_df = build_metric_grids(ca_df, cs_df, max_bin=29)
>>> speed_df.head()      # rows = bins, columns = datetimes
"""

from __future__ import annotations
from typing import Tuple, Sequence

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
def _nearest_ca_rows(
    ca_df: pd.DataFrame,                     # BIN0 only
    timeline: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """
    Return a CA-slice aligned to *timeline* (one row per timestamp),
    using the nearest-time rule.
    """
    # Make sure both sides are sorted for merge_asof
    ca_sorted       = ca_df.sort_values("datetime").reset_index(drop=True)
    timeline_df     = pd.DataFrame({"datetime": timeline}).sort_values("datetime")

    aligned = pd.merge_asof(
        timeline_df,
        ca_sorted,
        on="datetime",
        direction="nearest",      # pick earlier or later, whichever closer
    )
    # now aligned has *all* timeline rows, with columns from CA
    return aligned


# ------------------------------------------------------------------
def build_metric_grids(
    ca_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    max_bin: int = 29,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    ca_df : DataFrame
        Only BIN0 rows (from $PNORCA).
    cs_df : DataFrame
        BIN1 … BINn rows (from $PNORCS).
    max_bin : int
        Last bin to include (default 29 → BIN0 … BIN29).

    Returns
    -------
    speed_df, direction_df : DataFrame
        Rows  = BIN0 … BIN{max_bin}
        Cols  = unique CS timestamps, sorted ascending
        Cells = NaN for missing / invalid measurements
    """
    # -- 1) reference timeline comes from CS records ----------------
    timeline = (
        cs_df["datetime"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    # -- 2) build BIN1+ grids straight from CS ----------------------
    bins_requested = [f"BIN{i}" for i in range(1, max_bin + 1)]

    speed_cs = (
        cs_df
        .pivot_table(
            index="bin", columns="datetime", values="speed_ms"
        )
        .reindex(index=bins_requested)
    )

    dir_cs   = (
        cs_df
        .pivot_table(
            index="bin", columns="datetime", values="direction_deg"
        )
        .reindex(index=bins_requested)
    )

    # -- 3) align BIN0 (CA) to same timeline ------------------------
    ca_aligned = _nearest_ca_rows(ca_df, timeline)

    speed_bin0 = pd.Series(ca_aligned["speed_ms"].to_numpy(),
                           index=timeline, name="BIN0")
    dir_bin0   = pd.Series(ca_aligned["direction_deg"].to_numpy(),
                           index=timeline, name="BIN0")

    # -- 4) concatenate BIN0 row on top -----------------------------
    speed_df = pd.concat([speed_bin0.to_frame().T, speed_cs], axis=0)
    dir_df   = pd.concat([dir_bin0.to_frame().T,   dir_cs], axis=0)

    # ensure correct row order (BIN0 … BIN{max_bin})
    final_index = [f"BIN{i}" for i in range(0, max_bin + 1)]
    speed_df    = speed_df.reindex(index=final_index)
    dir_df      = dir_df.reindex(index=final_index)

    # columns already sorted because *timeline* is sorted
    return speed_df, dir_df

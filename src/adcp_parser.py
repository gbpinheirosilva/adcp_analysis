"""
adcp_parser.py
==============

Tiny helper for Nortek ADCP text logs (NMEA-like sentences):
    $PNORCA*  → mid-depth average (should be BIN0)
    $PNORCS*  → per-cell profiles  (BIN1 … BINn)

Functions
---------
parse_file(path)          -> (ca_df, cs_df)
make_fusion_df(ca, cs)    -> wide, time-sorted DataFrame for quick plots
save_tables_html(ca, cs)  -> two standalone HTML tables (optional CSS)
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

# ------------------------------------------------------------------
# Regex to capture sentence type and the optional trailing digits
# ------------------------------------------------------------------
_RE_SENTENCE = re.compile(r"^\$(PNORC[AS])(\d*),([^*]*)")

# Columns we keep (1-based positions 1,3,4,5,6,7  →  0-based indices below)
_KEEP_IDX = [0, 2, 3, 4, 5, 6]
_COLS     = ["bin", "date", "time", "depth_m", "speed_ms", "direction_deg"]

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _parse_line(line: str) -> tuple[str, list[str]] | None:
    """
    Return ('A' | 'S',  [fields…])  |  None  if line is irrelevant.

    ─  $PNORCA* → kind='A',  BIN always forced to '0'
    ─  $PNORCS17 → kind='S', BIN '17'
    """
    m = _RE_SENTENCE.match(line)
    if not m:
        return None

    kind_full, bin_num, rest = m.groups()
    kind = kind_full[-1]                       # 'A' or 'S'

    if kind == "A":                            # CA = BIN0 (ignore digits)
        bin_num = "0"

    fields = rest.split(",")                   # everything after first comma
    return kind, [bin_num] + fields            # prepend BIN

def _as_df(rows: list[list[str]]) -> pd.DataFrame:
    """Build DataFrame → keep wanted cols, cast numerics, mark invalids."""
    trimmed = []
    for r in rows:
        subset = [r[i] for i in _KEEP_IDX]
        subset[0] = f"BIN{subset[0]}"          # turn '0' → 'BIN0', etc.
        trimmed.append(subset)

    df = pd.DataFrame(trimmed, columns=_COLS)

    # Cast numeric columns
    df[["depth_m", "speed_ms", "direction_deg"]] = (
        df[["depth_m", "speed_ms", "direction_deg"]]
        .apply(pd.to_numeric, errors="coerce")
    )
    df.replace(-999.25, pd.NA, inplace=True)   # flag invalid measurements
    return df

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def parse_file(filepath: str | Path):
    """
    Parse a raw Nortek ADCP log.

    Returns
    -------
    ca_df : DataFrame  (only BIN0 rows from $PNORCA*)
    cs_df : DataFrame  (BIN1 … BINn rows from $PNORCS*)
    """
    ca_rows, cs_rows = [], []

    with open(filepath, "r", encoding="utf-8") as fh:
        for ln in fh:
            parsed = _parse_line(ln.strip())
            if not parsed:
                continue
            kind, fields = parsed
            (ca_rows if kind == "A" else cs_rows).append(fields)

    ca_df = _as_df(ca_rows)
    cs_df = _as_df(cs_rows)

    # Build a combined timestamp column once (will be useful downstream)
    for df in (ca_df, cs_df):
        df["datetime"] = pd.to_datetime(
            df["date"] + df["time"], format="%d%m%y%H%M%S"
        )

    return ca_df, cs_df


def make_fusion_df(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CA + CS into one wide table (index = datetime, columns = metric_BINx).

    Example
    -------
    depth_m_BIN0 | depth_m_BIN1 | … | speed_ms_BIN0 | speed_ms_BIN1 | …
    """
    big = pd.concat(dfs, ignore_index=True)
    wide = (
        big.pivot_table(
            index="datetime",
            columns="bin",
            values=["depth_m", "speed_ms", "direction_deg"],
        )
        .sort_index()
    )
    wide.columns = [f"{metric}_{bin_}" for metric, bin_ in wide.columns]
    return wide


def save_tables_html(
    ca_df: pd.DataFrame,
    cs_df: pd.DataFrame,
    out_dir: str | Path = "html",
    css_path: str | Path | None = None,
):
    """
    Dump two self-contained HTML files:  PNORCA.html  &  PNORCS.html

    They will optionally include  <link rel="stylesheet">  for your CSS.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for name, df in {"PNORCA": ca_df, "PNORCS": cs_df}.items():
        html = df.to_html(index=False, classes=["adcp", name.lower()])
        if css_path:
            rel = Path(css_path).name
            html = (
                f'<!doctype html><html><head>'
                f'<link rel="stylesheet" href="{rel}"></head><body>{html}</body></html>'
            )
        (out_dir / f"{name}.html").write_text(html, encoding="utf-8")

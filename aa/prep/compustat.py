"""
Compustat Annual (FUNDA) → clean panel (annual variables on a monthly index)

Rules (per AA protocol):
- Use FUNDA (annual) with identifiers: gvkey, datadate, fyear, fyr.
- Collapse duplicates within (gvkey, fyear) by keeping the latest datadate.
- Assign each annual observation to the month of **June in the year after datadate.year**,
  normalized to month-end (e.g., 2000-12-31 → assign 2001-06-30).
- Output is tidy and easy to link via CCM at the assign_month.

Returns a DataFrame with:
  ['gvkey','datadate','fyear','fyr','assign_month', <common numeric cols if present>]
"""

from __future__ import annotations

from typing import Iterable, List
import pandas as pd


_REQUIRED = ["gvkey", "datadate", "fyear", "fyr"]
_COMMON = ["at", "ceq", "sale", "lt", "seqq", "txditc"]  # included if present


def _ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=False, errors="coerce").dt.tz_localize(None)


def _june_yplus1_month_end(dates: pd.Series) -> pd.Series:
    y_plus_1 = dates.dt.year + 1
    june1 = pd.to_datetime(
        y_plus_1.astype(str) + "-06-01", format="%Y-%m-%d", errors="coerce"
    )
    return (june1 + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")


def prepare_compustat_annual(raw_funda: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FUNDA into an annual panel aligned to June Y+1 month-end.
    """
    df = raw_funda.copy()

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required FUNDA columns: {missing}")

    keep_cols: List[str] = [c for c in _REQUIRED + _COMMON if c in df.columns]
    extra_cols: List[str] = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + extra_cols]

    df["gvkey"] = df["gvkey"].astype(str)
    df["datadate"] = _ensure_datetime(df["datadate"])
    df["fyear"] = df["fyear"].astype("Int64")
    df["fyr"] = df["fyr"].astype("Int64")

    # Keep the latest datadate per (gvkey, fyear)
    df = (
        df.sort_values(["gvkey", "fyear", "datadate"])
        .groupby(["gvkey", "fyear"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    df["assign_month"] = _june_yplus1_month_end(df["datadate"])

    # numeric coercion for common fields, when present
    numeric_candidates: Iterable[str] = [c for c in _COMMON if c in df.columns]
    for c in numeric_candidates:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out_cols = ["gvkey", "datadate", "fyear", "fyr", "assign_month"] + list(
        numeric_candidates
    )
    return df[out_cols].sort_values(["gvkey", "assign_month"]).reset_index(drop=True)

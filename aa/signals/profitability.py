"""
Profitability anomaly signal.

Operating profitability, as defined by Novy‑Marx (2013), is earnings
before interest, taxes and extraordinary items minus cost of goods
sold, SG&A and interest expense, divided by book equity plus
minority interest.  The Fama–French profitability factor uses this
ratio measured for fiscal year *t−1* and available six months later
to form portfolios in June of year *t*【365329894510713†L44-L48】.  Firms with high
operating profitability tend to earn higher returns.

The function below computes a profitability signal from Compustat
fundamental data and aligns it to CRSP monthly dates with a six‑month
lag.  The caller must provide pre‑linked Compustat data containing
operating income components and book equity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_profitability_signal(
    crsp: pd.DataFrame, funda: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute operating profitability signals.

    Parameters
    ----------
    crsp : DataFrame
        Monthly CRSP data with columns ``date`` and ``permno``.
    funda : DataFrame
        Compustat fundamentals with columns ``permno``, ``datadate``,
        ``op`` (operating profitability numerator), ``be`` (book
        equity) and optionally ``mi`` (minority interest).  If
        ``mi`` is absent it is assumed to be zero.

    Returns
    -------
    DataFrame
        Columns ``date``, ``permno`` and ``signal`` where
        ``signal`` is ``op / (be + mi)`` using the most recent
        fundamental observation available at least six months prior to
        the CRSP date.  Missing inputs yield NaN signals.

    Notes
    -----
    Operating profitability signals are not negated; higher values
    correspond to more profitable firms.  The same reporting lag
    applied to book‑to‑market is used here【365329894510713†L44-L48】.
    """
    req_crsp = {"date", "permno"}
    req_funda = {"permno", "datadate", "op", "be"}
    miss_crsp = req_crsp - set(crsp.columns)
    miss_funda = req_funda - set(funda.columns)
    if miss_crsp:
        raise KeyError(f"crsp is missing required columns: {sorted(miss_crsp)}")
    if miss_funda:
        raise KeyError(f"funda is missing required columns: {sorted(miss_funda)}")

    # Normalise CRSP
    crsp_df = crsp.copy()
    crsp_df["date"] = pd.to_datetime(crsp_df["date"], errors="coerce").dt.tz_localize(
        None
    )
    crsp_df = crsp_df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Normalise fundamentals
    fdf = funda.copy()
    fdf["datadate"] = pd.to_datetime(fdf["datadate"], errors="coerce").dt.tz_localize(
        None
    )
    fdf[["op", "be"]] = fdf[["op", "be"]].apply(pd.to_numeric, errors="coerce")
    if "mi" in fdf.columns:
        fdf["mi"] = pd.to_numeric(fdf["mi"], errors="coerce")
    else:
        fdf["mi"] = 0.0
    fdf = fdf.sort_values(["permno", "datadate"]).reset_index(drop=True)
    # Compute profitability ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = fdf["be"] + fdf["mi"]
        fdf["prof"] = fdf["op"] / denom
    # Reporting lag
    fdf["report_date"] = fdf["datadate"] + pd.DateOffset(months=6)
    # Group‑wise as‑of merge to avoid sorting issues across multiple permnos.
    merged_parts = []
    for pno, crsp_sub in crsp_df.groupby("permno", sort=False):
        fsub = fdf[fdf["permno"] == pno].sort_values("report_date")
        if fsub.empty:
            tmp = crsp_sub[["date", "permno"]].copy()
            tmp["signal"] = np.nan
            merged_parts.append(tmp)
            continue
        crsp_sub_sorted = crsp_sub.sort_values("date")
        msub = pd.merge_asof(
            crsp_sub_sorted,
            fsub[["report_date", "prof"]],
            left_on="date",
            right_on="report_date",
            direction="backward",
            allow_exact_matches=False,
        )
        msub.rename(columns={"prof": "signal"}, inplace=True)
        merged_parts.append(msub[["date", "permno", "signal"]])
    merged = pd.concat(merged_parts, ignore_index=True)
    return merged[["date", "permno", "signal"]]

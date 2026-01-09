"""
Investment anomaly signal.

Investment, one of the five Fama–French factors, captures the
tendency for firms that invest less aggressively to earn higher
returns.  The investment ratio used to form portfolios in year *t* is
the change in total assets from the fiscal year ending in *t−2* to
the fiscal year ending in *t−1*, divided by the level of assets at
*t−2*【365329894510713†L50-L52】.  This module implements a monthly proxy of that
ratio with a six‑month reporting lag.

We assume that Compustat fundamentals have already been linked to
PERMNO and that each observation includes total assets (``at``) and
fiscal year end (``datadate``).  The caller must provide CRSP
monthly dates in order to align the annual signals to months.  The
resulting DataFrame can be passed directly to the portfolio sorting
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_investment_signal(crsp: pd.DataFrame, funda: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the investment signal from Compustat total assets.

    Parameters
    ----------
    crsp : DataFrame
        Monthly CRSP data with columns ``date`` and ``permno``.
    funda : DataFrame
        Clean Compustat data with columns ``permno``, ``datadate`` and
        ``at`` (total assets).  ``datadate`` should correspond to the
        fiscal year end.

    Returns
    -------
    DataFrame
        Columns ``date``, ``permno`` and ``signal`` where
        ``signal`` is defined as the negative of the year‑over‑year
        growth in total assets: ``-(at_t1 - at_t0)/at_t0``.  A
        six‑month reporting lag is applied as in the standard
        Fama–French construction【365329894510713†L50-L52】.  Missing data yield NaN
        signals.

    Notes
    -----
    The investment factor loads positively on firms that have invested
    aggressively (high asset growth) and negatively on firms that
    shrink their assets.  Sorting on the negative of asset growth
    therefore aligns high signals with low investment (value firms).
    """
    req_crsp = {"date", "permno"}
    req_funda = {"permno", "datadate", "at"}
    miss_c = req_crsp - set(crsp.columns)
    miss_f = req_funda - set(funda.columns)
    if miss_c:
        raise KeyError(f"crsp is missing required columns: {sorted(miss_c)}")
    if miss_f:
        raise KeyError(f"funda is missing required columns: {sorted(miss_f)}")

    # Prepare CRSP dates
    crsp_df = crsp.copy()
    crsp_df["date"] = pd.to_datetime(crsp_df["date"], errors="coerce").dt.tz_localize(
        None
    )
    crsp_df = crsp_df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Prepare fundamentals
    fdf = funda.copy()
    fdf["datadate"] = pd.to_datetime(fdf["datadate"], errors="coerce").dt.tz_localize(
        None
    )
    fdf["at"] = pd.to_numeric(fdf["at"], errors="coerce")
    fdf = fdf.sort_values(["permno", "datadate"]).reset_index(drop=True)
    # Compute year‑over‑year asset growth per permno
    fdf["at_lag"] = fdf.groupby("permno")["at"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdf["inv"] = (fdf["at"] - fdf["at_lag"]) / fdf["at_lag"]
    # Negative sign so that high signals correspond to low investment
    fdf["signal"] = -fdf["inv"]
    # Reporting lag of six months
    fdf["report_date"] = fdf["datadate"] + pd.DateOffset(months=6)
    # Drop rows where lag is missing
    fdf = fdf.dropna(subset=["report_date"])
    # Group‑wise as‑of merge: similar to book‑to‑market, perform the as‑of
    # join within each permno group to avoid sorting issues when data are
    # interleaved across permnos.
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
            fsub[["report_date", "signal"]],
            left_on="date",
            right_on="report_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(msub[["date", "permno", "signal"]])
    merged = pd.concat(merged_parts, ignore_index=True)
    return merged[["date", "permno", "signal"]]

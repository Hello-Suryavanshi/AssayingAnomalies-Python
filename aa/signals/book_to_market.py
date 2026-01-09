"""
Book-to-market anomaly signal.

The book-to-market (B/M) ratio is a classical value signal that compares
a firm's accounting book value to its market valuation.  In the
Fama–French construction, book equity for fiscal year *t−1* is
matched with market equity measured at the end of December of *t−1* to
form portfolios in June of year *t*.  The Python implementation here
follows this lag structure in a monthly setting by requiring that
book equity observations be at least six months old before they can
inform the signal.  A ``pd.merge_asof`` on a six-month reporting lag
aligns Compustat fundamentals to CRSP dates.

Because the raw Compustat GVKEYs have already been linked to PERMNO
outside this module, the function below expects the fundamentals
DataFrame to contain ``permno``, ``datadate`` (fiscal year end) and
``be`` (book equity).  Missing fundamentals or market equity lead to
missing signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_book_to_market_signal(
    crsp: pd.DataFrame, funda: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the log book-to-market ratio with a six-month reporting lag.

    Parameters
    ----------
    crsp : DataFrame
        Monthly CRSP data with at least columns ``date``, ``permno`` and
        ``me`` (market equity).
    funda : DataFrame
        Cleaned Compustat fundamentals with columns ``permno``,
        ``datadate`` (fiscal year end) and ``be`` (book equity).
        ``datadate`` should be a period end (e.g. fiscal year end) on
        which book equity is measured.  The ``permno`` linkage must be
        pre-computed by the caller (e.g. via a CCM link).

    Returns
    -------
    DataFrame
        Columns: ``date``, ``permno``, ``signal`` where ``signal`` is
        ``log(be/me)`` using the most recent book equity observation
        available at least six months prior to the CRSP date.  Rows
        where either book or market equity is missing result in NaN
        signals.
    """
    # Validate inputs
    req_crsp = {"date", "permno", "me"}
    req_funda = {"permno", "datadate", "be"}
    missing_crsp = req_crsp - set(crsp.columns)
    missing_funda = req_funda - set(funda.columns)
    if missing_crsp:
        raise KeyError(f"crsp is missing required columns: {sorted(missing_crsp)}")
    if missing_funda:
        raise KeyError(f"funda is missing required columns: {sorted(missing_funda)}")

    # Normalize and sort CRSP
    crsp_df = crsp.copy()
    crsp_df["date"] = pd.to_datetime(crsp_df["date"], errors="coerce").dt.tz_localize(
        None
    )
    crsp_df["me"] = pd.to_numeric(crsp_df["me"], errors="coerce")
    crsp_df = crsp_df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Normalize fundamentals
    funda_df = funda.copy()
    funda_df["datadate"] = pd.to_datetime(
        funda_df["datadate"], errors="coerce"
    ).dt.tz_localize(None)
    funda_df["be"] = pd.to_numeric(funda_df["be"], errors="coerce")
    funda_df = funda_df.dropna(subset=["permno", "datadate"])

    # Reporting lag: datadate becomes available at datadate + 6 months
    funda_df["report_date"] = funda_df["datadate"] + pd.DateOffset(months=6)
    funda_df = funda_df.sort_values(["permno", "report_date"]).reset_index(drop=True)

    # Group-wise as-of merge per permno to avoid monotonicity issues in fixtures
    merged_parts: list[pd.DataFrame] = []
    for pno, crsp_sub in crsp_df.groupby("permno", sort=False):
        funda_sub = funda_df[funda_df["permno"] == pno].sort_values("report_date")

        if funda_sub.empty:
            tmp = crsp_sub.copy()
            tmp["signal"] = np.nan
            merged_parts.append(tmp)
            continue

        # IMPORTANT: drop permno from right-hand side to avoid permno_x/permno_y
        # since permno is already fixed by filtering on pno
        funda_sub = funda_sub[["report_date", "be"]].copy()

        crsp_sub_sorted = crsp_sub.sort_values("date")
        merged_sub = pd.merge_asof(
            crsp_sub_sorted,
            funda_sub,
            left_on="date",
            right_on="report_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged_parts.append(merged_sub)

    merged = pd.concat(merged_parts, ignore_index=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["signal"] = np.log(merged["be"] / merged["me"])

    # permno now safely exists (from the left side)
    return merged[["date", "permno", "signal"]]

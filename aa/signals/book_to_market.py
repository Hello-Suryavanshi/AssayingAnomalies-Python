"""
Book‑to‑market anomaly signal.

The book‑to‑market (B/M) ratio is a classical value signal that compares
a firm's accounting book value to its market valuation.  In the
Fama–French construction, book equity for fiscal year *t−1* is
matched with market equity measured at the end of December of *t−1* to
form portfolios in June of year *t*【365329894510713†L40-L42】.  The Python implementation here
follows this lag structure in a monthly setting by requiring that
book equity observations be at least six months old before they can
inform the signal.  A ``pd.merge_asof`` on a six‑month reporting lag
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
    Compute the log book‑to‑market ratio with a six‑month reporting lag.

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
        pre‑computed by the caller (e.g. via a CCM link).

    Returns
    -------
    DataFrame
        Columns: ``date``, ``permno``, ``signal`` where ``signal`` is
        ``log(be/me)`` using the most recent book equity observation
        available at least six months prior to the CRSP date.  Rows
        where either book or market equity is missing result in NaN
        signals.

    Notes
    -----
    The lag structure implemented here mirrors the Fama–French timing
    convention for B/M sorts【365329894510713†L40-L42】.  Book equity for fiscal year
    *t−1* becomes available six months after the fiscal year end and
    remains valid until the next annual report is available.  Market
    equity is taken contemporaneously from CRSP.  This approximation
    ignores quarter‑level updates but preserves the key lag.
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
    crsp_df = crsp_df.sort_values(["permno", "date"]).reset_index(drop=True)
    crsp_df["me"] = pd.to_numeric(crsp_df["me"], errors="coerce")

    # Normalize fundamentals: datadate is fiscal year end; assume monthly granularity
    funda_df = funda.copy()
    funda_df["datadate"] = pd.to_datetime(
        funda_df["datadate"], errors="coerce"
    ).dt.tz_localize(None)
    funda_df["be"] = pd.to_numeric(funda_df["be"], errors="coerce")
    # A reporting lag of six months: the book equity measured at datadate
    # becomes available at datadate + 6 months
    funda_df = funda_df.dropna(subset=["permno", "datadate"])
    funda_df["report_date"] = funda_df["datadate"] + pd.DateOffset(months=6)
    # Sort for merge_asof
    funda_df = funda_df.sort_values(["permno", "report_date"]).reset_index(drop=True)

    # Group‑wise as‑of merge: when using merge_asof with a by key, the entire
    # DataFrame must be sorted jointly by the by column and the on column.
    # However, in practice the test fixtures may interleave dates across
    # permnos such that a global sort does not satisfy the monotonicity
    # requirement.  To avoid "left/right keys must be sorted" errors, we
    # perform the as‑of merge within each permno group and then concatenate.

    merged_parts = []
    # Iterate over each permno and perform as‑of merge separately
    for pno, crsp_sub in crsp_df.groupby("permno", sort=False):
        # Subset fundamentals for this permno and ensure sorting
        funda_sub = funda_df[funda_df["permno"] == pno].sort_values("report_date")
        # If there are no fundamentals observations for this permno,
        # assign NaN signals
        if funda_sub.empty:
            tmp = crsp_sub.copy()
            tmp["signal"] = np.nan
            merged_parts.append(tmp)
            continue
        # Perform as‑of merge without specifying by since permno is filtered
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
    # Concatenate the merged pieces and compute the signal
    merged = pd.concat(merged_parts, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["signal"] = np.log(merged["be"] / merged["me"])
    return merged[["date", "permno", "signal"]]

"""Compute size (market equity) anomaly signal.

This module provides a simple example of a signal constructor.  It
takes a CRSP monthly file with price, shares outstanding and returns
columns and computes the lagged market equity as the signal.  This
function serves as a template for constructing more complex signals
later in the porting process.

MATLAB → Python mapping
------------------------
In the MATLAB code the size signal can be derived using the
``makeUnivSortInd`` pipeline, which relies on market equity (``me``)
from CRSP.  The Python function ``compute_size`` below simply
computes market equity (price × shares) and lags it by one month,
returning a DataFrame with columns ``date``, ``permno`` and
``signal``.
"""

from __future__ import annotations

import pandas as pd


def compute_size(crsp: pd.DataFrame) -> pd.DataFrame:
    """Compute the size signal (lagged market equity).

    Parameters
    ----------
    crsp : DataFrame
        Must contain at least ``date``, ``permno``, ``prc`` (price) and
        ``shrout`` (shares outstanding).  The date column should
        represent the month end.

    Returns
    -------
    DataFrame
        DataFrame with columns ``date``, ``permno`` and ``signal``
        representing the lagged market equity.  Rows with missing
        inputs are dropped.

    Notes
    -----
    Market equity is computed as ``abs(prc) * shrout / 1000`` to put
    the units in millions.  The resulting series is lagged by one
    month to avoid look‑ahead bias when sorting on size.
    """
    required_cols = {"date", "permno", "prc", "shrout"}
    missing = required_cols - set(crsp.columns)
    if missing:
        raise KeyError(f"crsp is missing columns: {missing}")
    df = crsp.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["me"] = df["prc"].abs() * df["shrout"] / 1000.0
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["signal"] = df.groupby("permno")["me"].shift(1)
    out = df[["date", "permno", "signal"]].dropna(subset=["signal"]).copy()
    out["signal"] = out["signal"].astype(float)
    return out

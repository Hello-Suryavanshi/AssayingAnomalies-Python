"""
Momentum anomaly signal (12‑minus‑2).

Momentum strategies rank stocks by their past returns.  Carhart (1997)
popularised the 12‑minus‑2 specification, in which cumulative returns
from month *t−12* to *t−2* are used to predict returns in month *t*.
The Fama–French momentum factor is built using prior (2–12) returns
with NYSE breakpoints【696072703907323†L31-L37】.  This implementation computes
cross‑sectional momentum signals for each stock and month by forming
the cumulative gross return over the 11 months from *t−12* through
*t−2*, excluding the most recent month to avoid microstructure
effects【696072703907323†L31-L47】.

Given a DataFrame of monthly returns, the function
:func:`compute_momentum_signal` returns a DataFrame of signals that
can be passed to the portfolio sorting routines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_signal(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12‑minus‑2 momentum signals from CRSP returns.

    Parameters
    ----------
    crsp : DataFrame
        Must contain columns ``date``, ``permno`` and ``ret`` (one
        month raw return as a decimal).  Returns should include
        distributions (i.e. not be net of dividends).

    Returns
    -------
    DataFrame
        Columns ``date``, ``permno``, ``signal`` where ``signal`` is
        the cumulative return from month *t−12* through *t−2* (11
        monthly returns) expressed as a gross return minus one.  Dates
        where fewer than 11 lagged returns are available yield NaN.

    Notes
    -----
    The Carhart momentum specification excludes the most recent month
    (t−1) from the lookback window to mitigate short‑term reversal
    effects.  For each stock we compute the product of (1+ret) over
    the 11‑month window and subtract one to obtain the cumulative
    simple return.  Missing returns within the window propagate a
    missing signal.
    """
    required = {"date", "permno", "ret"}
    missing = required - set(crsp.columns)
    if missing:
        raise KeyError(f"crsp is missing required columns: {sorted(missing)}")
    df = crsp.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["gross"] = 1.0 + df["ret"]

    # Compute rolling product of 11 months excluding the most recent month (shift by 2)
    def _roll_prod(x: pd.Series) -> pd.Series:
        # Shift by 2 to exclude t and t−1, then rolling product over window=11
        shifted = x.shift(2)
        prod = shifted.rolling(window=11, min_periods=11).apply(np.prod, raw=True)
        return prod

    df["cumprod"] = df.groupby("permno")["gross"].transform(_roll_prod)
    with np.errstate(invalid="ignore"):
        df["signal"] = df["cumprod"] - 1.0
    return df[["date", "permno", "signal"]]

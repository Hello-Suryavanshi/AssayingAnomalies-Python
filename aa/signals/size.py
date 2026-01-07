"""
Construction of the size anomaly signal.

The size characteristic in the Assaying Anomalies framework is defined
as the negative logarithm of a firm's market equity.  To avoid look‑
ahead bias, market equity is lagged one month prior to computing the
signal.  This module implements a pure function that accepts CRSP
monthly data and returns a tidy DataFrame of signals.

MATLAB → Python mapping
------------------------
The MATLAB function ``makeSizeSignal`` computes ``size = -log(ME)`` on
lagged market equity.  The Python function ``compute_size_signal`` here
follows the same logic, returning a DataFrame with columns ``date``,
``permno`` and ``signal``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_size_signal(crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the lagged size signal from CRSP data.

    Parameters
    ----------
    crsp : DataFrame
        Must contain columns ``date``, ``permno`` and ``me`` (market equity).

    Returns
    -------
    DataFrame
        Columns: ``date``, ``permno``, ``signal`` where ``signal`` is
        ``-log(me_lag)`` and ``me_lag`` is the prior month market equity.

    Notes
    -----
    Rows for which ``me`` is missing in the prior month will have NaN
    signals.
    """
    required = {"date", "permno", "me"}
    missing = required - set(crsp.columns)
    if missing:
        raise KeyError(f"crsp is missing required columns: {sorted(missing)}")
    # sort to ensure lagging is correct
    df = crsp.copy().sort_values(["permno", "date"]).reset_index(drop=True)
    df["me"] = pd.to_numeric(df["me"], errors="coerce")
    # lag ME by one observation per permno
    df["me_lag"] = df.groupby("permno")["me"].shift(1)
    # compute signal: negative log of lagged ME
    df["signal"] = -np.log(df["me_lag"].astype(float))
    return df[["date", "permno", "signal"]]

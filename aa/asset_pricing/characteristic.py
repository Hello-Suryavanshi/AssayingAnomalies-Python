r"""
Characteristic-managed portfolios.

This module implements linear characteristic-managed portfolio returns. A
characteristic-managed portfolio (CMP) uses lagged firm characteristics
as portfolio weights to construct long–short factors. The idea stems
from Daniel and Titman (1997) and subsequent literature that argues
characteristics provide better estimates of expected returns than do
factor loadings.

In practice, a characteristic-managed return at time ``t`` for
characteristic ``X`` is the cross-sectional covariance of lagged
characteristic values with next month’s returns. Formally, let
``X_{i,t}`` denote the lagged value of a characteristic for firm ``i``
observed at time ``t`` and ``R_{i,t+1}`` the return from ``t`` to ``t+1``.
The characteristic-managed payoff is

.. math::

    r_{t+1}^{\mathrm{CMP}} = \frac{1}{N_t}\sum_{i=1}^{N_t}
      \left( \tilde{X}_{i,t} \times R_{i,t+1} \right),

where ``\tilde{X}_{i,t} = (X_{i,t} - \bar{X}_t) / s_t`` is the
cross-sectional standardised characteristic (demeaned and scaled by its
standard deviation) and ``N_t`` is the number of stocks available at
time ``t``. Standardising ensures that CMP returns are scale invariant.

References
----------
Daniel, K., and S. Titman, 1997, Evidence on the characteristics of
  cross-sectional variation in stock returns. *Journal of Finance* 52, 1–33.

Kelly, B., S. Pruitt and Y. Su, 2019, Characteristics are covariances.
  *Journal of Financial Economics* 134, 501–524.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = ["characteristic_managed_portfolio"]


def characteristic_managed_portfolio(
    returns: pd.DataFrame,
    characteristics: pd.DataFrame,
    *,
    standardise: bool = True,
) -> pd.DataFrame:
    """
    Compute characteristic-managed portfolio returns for one or more characteristics.

    Parameters
    ----------
    returns : DataFrame
        Monthly stock returns with columns ``date``, ``permno`` and ``ret``.
        ``ret`` should be expressed in decimal form (e.g. 0.01 for 1%).
    characteristics : DataFrame
        Lagged firm characteristics with columns ``date`` and ``permno`` followed by
        one or more characteristic columns. Each characteristic is assumed to be
        observed at ``date`` and used to predict the return in the next month.
    standardise : bool, default True
        If True (recommended), demean and divide each characteristic by its
        cross-sectional standard deviation within each month before forming CMP
        returns. If False, demean but do not divide by the standard deviation.

    Returns
    -------
    DataFrame
        Wide DataFrame indexed by ``date`` with one column per characteristic.
    """
    if not {"date", "permno", "ret"}.issubset(returns.columns):
        raise KeyError("returns must contain columns 'date', 'permno' and 'ret'")
    if not {"date", "permno"}.issubset(characteristics.columns):
        raise KeyError(
            "characteristics must contain at least columns 'date' and 'permno'"
        )

    char_cols = [c for c in characteristics.columns if c not in {"date", "permno"}]
    if not char_cols:
        raise ValueError(
            "characteristics must contain at least one characteristic column"
        )

    # Normalise dates
    for df in (returns, characteristics):
        df.loc[:, "date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(
            None
        )

    data = (
        returns[["date", "permno", "ret"]]
        .merge(
            characteristics[["date", "permno"] + char_cols],
            on=["date", "permno"],
            how="inner",
        )
        .dropna(subset=["ret"] + char_cols)
    )

    out_rows: list[dict[str, Any]] = []

    for dt, grp in data.groupby("date", sort=True):
        # mypy-safe: dt might be inferred too broadly; cast explicitly
        dt_ts = pd.Timestamp(cast(Any, dt))

        ret_vec = grp["ret"].astype(float).to_numpy()
        row: dict[str, Any] = {"date": dt_ts}

        for col in char_cols:
            x = grp[col].astype(float).to_numpy()
            x = x - np.nanmean(x)

            if standardise:
                std = np.nanstd(x)
                if std > 0:
                    x = x / std

            row[col] = float(np.nanmean(x * ret_vec)) if len(x) else np.nan

        out_rows.append(row)

    return pd.DataFrame(out_rows)

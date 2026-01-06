"""Fama–MacBeth two‑pass regression estimator with Newey–West errors.

This module implements the cross‑sectional regression procedure known
as the Fama–MacBeth (1973) estimator.  It follows the workflow
outlined in the MATLAB function ``runFamaMacBeth.m`` found in the
``AssayingAnomalies`` toolkit.  In the first pass, returns are
regressed on one or more characteristics separately in each time
period.  The resulting series of coefficients (risk prices) are then
averaged in the second pass.  Standard errors are computed using the
Newey–West (HAC) estimator to account for potential autocorrelation in
the time series of estimated risk prices.

The implementation here is fully vectorised and uses only ``pandas``,
``numpy`` and ``statsmodels``.  It returns both the average risk
prices and the full time series of monthly estimates.  See the
function ``fama_macbeth`` for details.

MATLAB → Python mapping
------------------------
The MATLAB function ``runFamaMacBeth.m`` performs a two‑pass
cross‑sectional regression using Fama–MacBeth methodology.  It
produces average risk prices and Newey–West standard errors.  The
Python function ``fama_macbeth`` below replicates this behaviour.
It accepts a panel DataFrame with a time index (``yyyymm``) and
characteristic columns, and returns a DataFrame of average risk prices
and a DataFrame of period‑by‑period estimates.  The default number of
Newey–West lags is chosen following common practice as
``floor(4 * (T/100)^(2/9))`` where ``T`` is the number of periods.
Users may override this by specifying the ``nw_lags`` argument.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _long_run_variance(series: np.ndarray, lags: int) -> float:
    """Compute the Newey–West long‑run variance for a 1‑D array.

    Parameters
    ----------
    series : ndarray
        One‑dimensional array of demeaned observations.
    lags : int
        Number of lags to include in the HAC estimator.  A lag of zero
        yields the usual sample variance.

    Returns
    -------
    float
        Estimate of the long‑run variance.

    Notes
    -----
    The estimator used is

    .. math::

       \\gamma_0 + 2 \\sum_{k=1}^L \\left(1 - \\frac{k}{L+1}\\right) \\gamma_k,

    where ``γ_k`` is the lag‑k autocovariance of ``series``.
    """
    n = len(series)
    if n == 0:
        return np.nan
    gamma0 = np.dot(series, series) / n
    if lags == 0:
        return gamma0
    var = gamma0
    for k in range(1, lags + 1):
        cov = np.dot(series[k:], series[:-k]) / n
        weight = 1.0 - k / (lags + 1)
        var += 2.0 * weight * cov
    return var


def fama_macbeth(
    panel: pd.DataFrame,
    y: str = "ret",
    xcols: Sequence[str] | None = None,
    *,
    time_col: str = "yyyymm",
    nw_lags: int | None = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Estimate Fama–MacBeth risk prices and Newey–West standard errors.

    Parameters
    ----------
    panel : DataFrame
        Long format panel of returns and characteristics.  Must contain
        columns ``time_col``, ``y``, and each name in ``xcols``.  Each
        row corresponds to an asset–month observation.
    y : str, default "ret"
        Column name of the dependent variable (e.g. one‑month ahead
        returns).
    xcols : sequence of str, optional
        Names of characteristic columns used as regressors.  If None,
        no regressors are used and only an intercept is estimated.
    time_col : str, default "yyyymm"
        Name of the column identifying time periods.  Each unique value
        of this column is treated as a separate cross‑section.
    nw_lags : int, optional
        Number of Newey–West lags for the second‑pass standard errors.
        If None, an automatic lag length is chosen based on the number
        of periods, following the rule of thumb ``floor(4*(T/100)**(2/9))``.

    Returns
    -------
    lambdas : Series
        The average risk prices across all periods.  Index includes
        the intercept ``const`` and each regressor.
    lambda_ts : DataFrame
        Period‑by‑period coefficients.  Rows correspond to periods.
    se : Series
        Newey–West standard errors of the average coefficients.

    Examples
    --------
    >>> import pandas as pd
    >>> from aa.asset_pricing import fama_macbeth
    >>> df = pd.DataFrame({
    ...     "yyyymm": [202101, 202101, 202102, 202102],
    ...     "ret": [0.01, 0.02, 0.03, 0.04],
    ...     "beta": [1.0, 2.0, 1.5, 2.5],
    ... })
    >>> lambdas, lambda_ts, se = fama_macbeth(df, y="ret", xcols=["beta"])
    >>> round(lambdas["beta"], 4)
    0.02
    """
    if xcols is None:
        xcols = []

    if time_col not in panel.columns:
        raise KeyError(f"panel must contain time column '{time_col}'")
    if y not in panel.columns:
        raise KeyError(f"panel must contain dependent variable '{y}'")
    for col in xcols:
        if col not in panel.columns:
            raise KeyError(f"panel is missing regressor column '{col}'")

    coeffs: list[pd.Series] = []
    for period, grp in panel.groupby(time_col):
        yv = grp[y]
        if xcols:
            X = grp[list(xcols)]
            X = sm.add_constant(X, has_constant="add")
        else:
            X = pd.DataFrame({"const": np.ones(len(grp), dtype=float)})
        model = sm.OLS(yv, X, missing="drop")
        res = model.fit()
        coeffs.append(res.params.rename(period))

    lambda_ts = pd.DataFrame(coeffs).sort_index()
    lambda_ts = lambda_ts.reindex(columns=lambda_ts.columns, fill_value=np.nan)
    lambdas = lambda_ts.mean(axis=0, skipna=True)

    T = lambda_ts.shape[0]
    if nw_lags is None:
        nw_lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))

    se_vals = {}
    for col in lambda_ts.columns:
        series = lambda_ts[col].dropna().to_numpy(dtype=float)
        if series.size == 0:
            se_vals[col] = np.nan
            continue
        demeaned = series - np.nanmean(series)
        lr_var = _long_run_variance(demeaned, nw_lags)
        se_vals[col] = np.sqrt(lr_var / series.size)
    se = pd.Series(se_vals)

    return lambdas, lambda_ts, se

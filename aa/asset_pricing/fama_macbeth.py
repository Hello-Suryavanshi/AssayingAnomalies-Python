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
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["fama_macbeth"]


def _long_run_variance(series: np.ndarray, lags: int) -> float:
    r"""Compute the Newey–West long-run variance for a 1-D array.
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

       \gamma_0 + 2 \sum_{k=1}^L \left(1 - \frac{k}{L+1}\right) \gamma_k,

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
    """
    Estimate Fama–MacBeth risk prices and Newey–West standard errors.

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
        Newey‑West standard errors of the average coefficients.
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
    # We iterate through periods and handle missing values explicitly.  If
    # a given period has no valid observations for the specified
    # regressors, we record NaNs for that period.  This avoids
    # situations where statsmodels raises a ValueError on an empty
    # design matrix.
    for period, grp in panel.groupby(time_col):
        # Build mask of rows with non‑missing y and all xcols
        mask = grp[y].notna()
        for col in xcols:
            mask &= grp[col].notna()
        grp_valid = grp.loc[mask]
        # Determine design matrix
        if xcols:
            # If no valid rows, record NaNs and continue
            if grp_valid.empty:
                coeff = pd.Series(
                    {"const": np.nan, **{c: np.nan for c in xcols}}, name=period
                )
                coeffs.append(coeff)
                continue
            yv = grp_valid[y]
            X = grp_valid[list(xcols)]
            X = sm.add_constant(X, has_constant="add")
        else:
            # No regressors, just intercept; drop missing y
            if grp_valid.empty:
                coeffs.append(pd.Series({"const": np.nan}, name=period))
                continue
            yv = grp_valid[y]
            X = pd.DataFrame(
                {"const": np.ones(len(grp_valid), dtype=float)}, index=grp_valid.index
            )
        # Fit OLS without automatically dropping more missing, since we've filtered
        try:
            model = sm.OLS(yv, X)
            res = model.fit()
            coeff = res.params.rename(period)
        except Exception:
            # If regression fails for any reason, record NaNs
            coeff = pd.Series({col: np.nan for col in X.columns}, name=period)
        # Ensure all expected columns are present
        for col in ["const"] + list(xcols):
            if col not in coeff.index:
                coeff.loc[col] = np.nan
        coeffs.append(coeff)

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

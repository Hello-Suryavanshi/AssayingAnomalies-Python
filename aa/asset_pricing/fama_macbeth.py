"""
Fama–MacBeth two‑pass regression estimators.

This module provides implementations of the cross‑sectional regression
procedures commonly used in asset pricing.  The basic estimator,
known as the Fama–MacBeth (1973) two‑pass method, fits a cross‑sectional
regression in each time period and then averages the resulting series
of coefficients.  Newey–West heteroskedasticity and autocorrelation
consistent (HAC) standard errors are used to account for serial
correlation in the estimated risk prices.

Two functions are provided:

``fama_macbeth``
    Replicates the original behaviour from earlier milestones.  It
    returns average coefficients, the full time series of
    period‑by‑period coefficients and their Newey–West standard errors.

``fama_macbeth_full``
    Extends the basic estimator to include t‑statistics and the
    number of cross‑sectional observation periods used for each
    coefficient.  This function is the recommended interface for
    multi‑factor regressions in milestone 3.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["fama_macbeth", "fama_macbeth_full"]


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

    where :math:`\gamma_k` is the lag‑k autocovariance of ``series``.
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
        Long format panel of returns and characteristics.  Must
        contain columns ``time_col``, ``y`` and each name in ``xcols``.
        Each row corresponds to an asset–month observation.
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
        of periods.

    Returns
    -------
    lambdas : Series
        The average risk prices across all periods.  Index includes
        the intercept ``const`` and each regressor.
    lambda_ts : DataFrame
        Period‑by‑period coefficients.  Rows correspond to periods.
    se : Series
        Newey–West standard errors of the average coefficients.

    Notes
    -----
    This function is retained for backward compatibility.  For
    additional statistics such as t‑values and observation counts, use
    :func:`fama_macbeth_full`.
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
        mask = grp[y].notna()
        for col in xcols:
            mask &= grp[col].notna()
        grp_valid = grp.loc[mask]
        if xcols:
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
            if grp_valid.empty:
                coeffs.append(pd.Series({"const": np.nan}, name=period))
                continue
            yv = grp_valid[y]
            X = pd.DataFrame(
                {"const": np.ones(len(grp_valid), dtype=float)}, index=grp_valid.index
            )
        try:
            model = sm.OLS(yv, X)
            res = model.fit()
            coeff = res.params.rename(period)
        except Exception:
            coeff = pd.Series({col: np.nan for col in X.columns}, name=period)
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


def fama_macbeth_full(
    panel: pd.DataFrame,
    y: str = "ret",
    xcols: Sequence[str] | None = None,
    *,
    time_col: str = "yyyymm",
    nw_lags: int | None = None,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Extended Fama–MacBeth regression with t‑statistics and counts.

    This function wraps :func:`fama_macbeth` but also computes
    t‑statistics (coefficients divided by their standard errors) and
    the number of cross‑sectional periods used to estimate each
    coefficient.  Observation counts are based on the number of
    non‑missing estimates in ``lambda_ts``.

    Returns a dictionary with keys ``lambdas`` (Series),
    ``lambda_ts`` (DataFrame), ``se`` (Series), ``tstat`` (Series) and
    ``n_obs`` (Series).
    """
    lambdas, lambda_ts, se = fama_macbeth(
        panel, y=y, xcols=xcols, time_col=time_col, nw_lags=nw_lags
    )
    # t‑statistics
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = lambdas / se
    # counts: number of periods with non‑missing coefficient
    n_obs = lambda_ts.notna().sum(axis=0)
    return {
        "lambdas": lambdas,
        "lambda_ts": lambda_ts,
        "se": se,
        "tstat": tstat,
        "n_obs": n_obs,
    }

"""
Diagnostics and stability tests for empirical asset pricing models.

This module provides a collection of functions to assess the temporal
stability of portfolio returns and regression coefficients.  Modern
factor research emphasises not only point estimates but also their
behaviour over time—rolling alphas, structural break tests and
parameter instability measures shed light on whether an anomaly is
consistent across market regimes or concentrated in particular
episodes.  The routines herein are deliberately lightweight and
depend only on `pandas`, `numpy`, `statsmodels` and `scipy`, all
available in this environment.

Functions
---------
rolling_mean(series, window)
    Compute a simple moving average of a time series.

rolling_regression(y, X, window)
    Estimate rolling ordinary least squares (OLS) regressions and return
    a DataFrame of coefficient paths.

chow_test(y, X, break_index)
    Perform the Chow test for a structural break at a known index.
    Returns the F‑statistic and p‑value based on the formula in
    【581989950563653†L165-L186】.  Assumes homoskedastic normal errors.

subsample_fama_macbeth(panel, break_time, y, xcols, time_col)
    Run Fama–MacBeth regressions on two subsamples split at a given
    time and return the coefficients, standard errors and t‑tests on
    differences.

cusum_test(coef_ts)
    Compute a simple CUSUM statistic for a sequence of regression
    coefficients.  The statistic is the maximum absolute cumulative sum
    of demeaned coefficients divided by the sample standard deviation.
    Large values signal parameter instability.

"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # type: ignore[import-not-found]

from .asset_pricing.fama_macbeth import fama_macbeth_full

__all__ = [
    "rolling_mean",
    "rolling_regression",
    "chow_test",
    "subsample_fama_macbeth",
    "cusum_test",
]


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple moving average.

    Parameters
    ----------
    series : Series
        A time‑indexed series of numeric values.
    window : int
        Number of consecutive observations to include in each average.

    Returns
    -------
    Series
        Rolling mean with the same index as ``series``.  The first
        ``window-1`` values will be ``NaN``.
    """
    return series.rolling(window=window, min_periods=window).mean()


def rolling_regression(
    y: pd.Series,
    X: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Estimate rolling OLS regressions.

    For each rolling window of length ``window`` over the observations
    in ``y`` and ``X``, fit an OLS regression of ``y`` on ``X`` and
    return the estimated coefficients.  The index of the returned
    DataFrame corresponds to the end of each window.

    Parameters
    ----------
    y : Series
        Dependent variable.  Must be aligned with ``X``.
    X : DataFrame
        Regressors.  Will be augmented with a constant term.
    window : int
        Length of the rolling window.

    Returns
    -------
    DataFrame
        Columns are regression coefficients (including ``const``).  The
        index labels are the endpoints of each window.
    """
    if not len(y) == len(X):
        raise ValueError("y and X must have the same length")
    coeffs: list[pd.Series] = []
    idx: list = []
    for start in range(0, len(y) - window + 1):
        end = start + window
        y_win = y.iloc[start:end]
        X_win = X.iloc[start:end]
        Xc = sm.add_constant(X_win, has_constant="add")
        try:
            model = sm.OLS(y_win, Xc)
            res = model.fit()
            coeffs.append(res.params)
        except Exception:
            coeffs.append(pd.Series({c: np.nan for c in Xc.columns}))
        idx.append(y.index[end - 1])
    return pd.DataFrame(coeffs, index=idx)


def chow_test(
    y: Sequence[float] | pd.Series,
    X: Sequence[Sequence[float]] | pd.DataFrame,
    break_index: int,
) -> Tuple[float, float]:
    """Perform the Chow test for a structural break at a known breakpoint.

    The Chow test compares the fit of a linear regression estimated on
    the full sample against the fits on two subsamples split at the
    specified breakpoint.  Under the null hypothesis that the
    regression coefficients are identical across the subsamples, the
    test statistic follows an F‑distribution with ``k`` and
    ``n1 + n2 - 2k`` degrees of freedom, where ``k`` is the number of
    parameters (including the intercept) and ``n1`` and ``n2`` are the
    sizes of the two subsamples【581989950563653†L165-L186】.

    Parameters
    ----------
    y : array‑like
        Dependent variable.  Should be a one‑dimensional sequence.
    X : array‑like
        Two‑dimensional array of regressors.  Must have the same number
        of rows as ``y``.  A constant term is added automatically.
    break_index : int
        Index at which to split the sample.  The first ``break_index``
        observations form the first subsample and the remainder form
        the second.  Must satisfy ``0 < break_index < n``.

    Returns
    -------
    f_stat : float
        Chow test statistic.
    p_value : float
        Right‑tail p‑value from the F distribution.
    """
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)

    n = len(y_arr)
    k = X_arr.shape[1]

    if not (0 < break_index < n):
        raise ValueError("break_index must lie strictly between 0 and n")
    # Add constant
    Xc = sm.add_constant(X, has_constant="add")
    k = Xc.shape[1]
    # Fit combined model
    model_full = sm.OLS(y, Xc).fit()
    sse_full = np.sum(model_full.resid**2)
    # Fit submodels
    model_1 = sm.OLS(y[:break_index], Xc[:break_index]).fit()
    model_2 = sm.OLS(y[break_index:], Xc[break_index:]).fit()
    sse_1 = np.sum(model_1.resid**2)
    sse_2 = np.sum(model_2.resid**2)
    n1 = break_index
    n2 = n - break_index
    # Compute Chow statistic (formula from 【581989950563653†L165-L186】)
    num = (sse_full - (sse_1 + sse_2)) / k
    den = (sse_1 + sse_2) / (n1 + n2 - 2 * k)
    f_stat = num / den if den != 0 else np.inf
    p_value = (
        1.0 - stats.f.cdf(f_stat, k, n1 + n2 - 2 * k) if np.isfinite(f_stat) else 0.0
    )
    return float(f_stat), float(p_value)


def subsample_fama_macbeth(
    panel: pd.DataFrame,
    break_time: Any,
    *,
    y: str = "ret",
    xcols: Sequence[str] | None = None,
    time_col: str = "yyyymm",
) -> Dict[str, Any]:
    """Estimate Fama–MacBeth regressions on two subsamples and test differences.

    Splits the input panel into two subsets using ``break_time`` and
    runs :func:`aa.asset_pricing.fama_macbeth_full` on each.  The
    difference between the average risk prices in the two subsamples is
    reported along with a t‑statistic computed using the pooled
    Newey–West variances.

    Parameters
    ----------
    panel : DataFrame
        Panel containing returns and regressors.  Must include columns
        ``time_col``, ``y`` and each name in ``xcols``.
    break_time : scalar
        Value of ``time_col`` at which to split the panel.  Observations
        with ``time_col`` strictly less than ``break_time`` form the
        first subsample; those with values greater than or equal to
        ``break_time`` form the second.  The break itself belongs to
        the second subsample.
    y : str, default 'ret'
        Name of the dependent variable.
    xcols : sequence of str or None, optional
        Names of regressors.  If None, only an intercept is estimated.
    time_col : str, default 'yyyymm'
        Column identifying cross‑sections.

    Returns
    -------
    dict
        ``sub1`` and ``sub2`` keys contain the outputs of
        :func:`aa.asset_pricing.fama_macbeth_full` on the two subsamples.
        ``diff`` contains the difference in average coefficients
        (sub2 minus sub1).  ``tstat`` contains t‑statistics for the
        difference computed as (λ2 – λ1) / sqrt(se1² + se2²).
    """
    if xcols is None:
        xcols = []
    # Split panel
    sub1 = panel[panel[time_col] < break_time]
    sub2 = panel[panel[time_col] >= break_time]
    results: Dict[str, Any] = {}
    fm1 = fama_macbeth_full(sub1, y=y, xcols=xcols, time_col=time_col)
    fm2 = fama_macbeth_full(sub2, y=y, xcols=xcols, time_col=time_col)
    results["sub1"] = fm1
    results["sub2"] = fm2
    # Align coefficient names
    lambdas1, lambdas2 = fm1["lambdas"], fm2["lambdas"]
    se1, se2 = fm1["se"], fm2["se"]
    # Compute differences and pooled standard errors
    diff = lambdas2 - lambdas1
    se_pooled = np.sqrt(se1**2 + se2**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = diff / se_pooled
    results["diff"] = diff
    results["tstat"] = tstat
    return results


def cusum_test(coef_ts: pd.Series) -> float:
    """Compute a simple CUSUM statistic for a sequence of coefficients.

    The CUSUM statistic is defined as the maximum absolute value of the
    cumulative sum of demeaned coefficients, scaled by their sample
    standard deviation.  Large values indicate instability of the mean
    over time.  This implementation does not compute p‑values but
    provides a diagnostic measure that can be compared across
    coefficients.

    Parameters
    ----------
    coef_ts : Series
        Time series of regression coefficients or portfolio returns.

    Returns
    -------
    float
        The CUSUM statistic (unitless).
    """
    x = coef_ts.to_numpy(dtype=float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return float("nan")
    x = x - np.nanmean(x)
    # Standard deviation
    sd = np.nanstd(x, ddof=0)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    # Cumulative sum
    cumsum = np.nancumsum(x / sd)
    return float(np.nanmax(np.abs(cumsum)))

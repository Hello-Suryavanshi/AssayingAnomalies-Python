"""
Standardised robustness checks for anomaly analyses.

This module defines a small framework for running a given analysis
under a grid of alternative specifications.  It encapsulates common
robustness choices in empirical asset pricing—such as equal‑ versus
value‑weighted returns, NYSE versus all‑stock breakpoints, winsorising
extreme signals and lagging predictors—and applies them in a
consistent manner.  By summarising the resulting outputs, the user
can quickly assess how sensitive a purported anomaly is to reasonable
perturbations of the research design.

The primary entry point is :func:`run_robustness_checks`, which
iterates over a sequence of :class:`RobustnessConfig` objects and
returns a tidy table of metrics.  An optional ``metric_fn`` can be
provided to extract a scalar quantity (e.g. a high–low spread or a
regression coefficient) from the full analysis output.  If no
``metric_fn`` is supplied, the full result of each run is stored
verbatim in the returned DataFrame.

Examples
--------
>>> from aa.asset_pricing.univariate import univariate_sort, SortConfig
>>> from aa.robustness import RobustnessConfig, run_robustness_checks
>>> import pandas as pd
>>> returns = pd.DataFrame({"date": pd.date_range("2020-01-31", periods=4, freq="M").repeat(3),
...                        "permno": [1, 2, 3] * 4,
...                        "ret": [0.01, 0.02, 0.03, -0.01, 0.04, 0.02, 0.03, -0.02, 0.01, 0.02, 0.05, -0.03]})
>>> signal = returns[["date", "permno"]].copy(); signal["signal"] = [0.1, 0.2, 0.3] * 4
>>> base_cfg = SortConfig(n_bins=2)
>>> configs = [
...     RobustnessConfig(weighting="EW", nyse_breaks=False, winsorize=None, lag=0),
...     RobustnessConfig(weighting="VW", nyse_breaks=False, winsorize=None, lag=0),
... ]
>>> def hl_metric(res):
...     # extract high–low EW spread from univariate_sort output
...     summ = res["summary"]
...     ls_row = summ.loc[summ["bin"] == "L‑S", "ret_ew"]
...     return float(ls_row.iloc[0]) if not ls_row.empty else float("nan")
>>> table = run_robustness_checks(
...     analysis_fn=univariate_sort,
...     base_config=base_cfg,
...     config_list=configs,
...     metric_fn=hl_metric,
...     returns=returns,
...     signal=signal,
...     size=None,
...     exch=None,
... )
>>> list(table.columns)
['weighting', 'nyse_breaks', 'winsorize', 'lag', 'holding_period', 'metric']

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, List, Optional, Sequence

import pandas as pd

__all__ = ["RobustnessConfig", "run_robustness_checks"]


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for a single robustness specification.

    Parameters
    ----------
    weighting : {'EW', 'VW'}, default 'EW'
        Whether to compute equal‑weighted (EW) or value‑weighted (VW) portfolio
        returns.  For analyses where the choice of weighting is implicit in
        the presence of a market capitalisation column (``me``) this flag
        controls whether the ``size`` DataFrame is passed to the analysis.
    nyse_breaks : bool, default False
        If True, breakpoints for portfolio sorts are computed using only
        NYSE stocks (``exchcd == 1``).  If False, the entire universe is
        used.
    winsorize : float or tuple of float or None, optional
        Quantile threshold(s) for winsorising the ``signal`` columns.  If a
        single float ``q`` is provided, values below the ``q``th percentile
        and above the ``1‑q``th percentile are truncated.  If a tuple
        ``(low, high)`` is provided, values below the ``low``th percentile
        and above the ``high``th percentile are truncated.  If None, no
        winsorisation is applied.
    lag : int, default 0
        Number of periods by which to lag the signal relative to returns.
        A lag of 0 uses contemporaneous signals; a lag of 1 uses the
        previous period's signals when forming portfolios today.  Signals are
        shifted within each ``permno`` (security) by this amount and
        observations without sufficient history are dropped.
    holding_period : int, default 1
        Intended to support alternative holding periods for signals.  In
        milestone 4 this parameter is retained for completeness but no
        transformation is applied; it will be available for future
        extensions.
    """

    weighting: str = "EW"
    nyse_breaks: bool = False
    winsorize: Optional[float | tuple[float, float]] = None
    lag: int = 0
    holding_period: int = 1


def _winsorize_series(s: pd.Series, limits: Sequence[float]) -> pd.Series:
    """Winsorise a pandas Series by quantile limits.

    Values below the lower quantile are set to that quantile; values above
    the upper quantile are set to the upper quantile.  Missing values are
    returned unchanged.  If the quantiles cannot be computed (e.g. empty
    series) the input is returned.

    Parameters
    ----------
    s : Series
        Series of numeric values.
    limits : sequence of float
        Two quantiles ``(low, high)`` between 0 and 1.  For example,
        ``(0.01, 0.99)`` will cap the bottom 1 % and top 1 % of the data.

    Returns
    -------
    Series
        Winsorised series.
    """
    s = s.astype(float)
    if s.dropna().empty:
        return s
    low, high = float(limits[0]), float(limits[1])
    q_low = s.quantile(low)
    q_high = s.quantile(high)
    return s.clip(lower=q_low, upper=q_high)


def _apply_lag(
    signal_df: pd.DataFrame, lag: int, date_col: str, permno_col: str
) -> pd.DataFrame:
    """Shift signal values by ``lag`` periods within each security.

    Observations without sufficient lag history are removed.  Dates are
    preserved; the shifted signals refer to past periods.  This helper
    assumes ``signal_df`` contains at least the columns specified by
    ``date_col``, ``permno_col`` and ``signal``.
    """
    if lag == 0:
        return signal_df.copy()
    if lag < 0:
        raise ValueError("lag must be non‑negative")
    df = signal_df.copy()
    df.sort_values([permno_col, date_col], inplace=True)
    df["signal"] = df.groupby(permno_col)["signal"].shift(lag)
    df = df.dropna(subset=["signal"])
    return df


def run_robustness_checks(
    *,
    analysis_fn: Callable[..., Any],
    base_config: Any,
    config_list: Iterable[RobustnessConfig],
    config_param: str = "config",
    metric_fn: Optional[Callable[[Any], Any]] = None,
    date_col: str = "date",
    permno_col: str = "permno",
    **data_kwargs: Any,
) -> pd.DataFrame:
    """Run an analysis function under a series of robustness configurations.

    Parameters
    ----------
    analysis_fn : callable
        The function implementing the empirical analysis.  It must accept
        the keyword argument named ``config_param`` (default ``"config"``)
        which will be set to the modified version of ``base_config`` for
        each robustness run.  Any additional keyword arguments provided
        through ``data_kwargs`` are forwarded unchanged except for
        signal‑specific transformations described below.
    base_config : object
        Baseline configuration object for the analysis function.  For
        portfolio sorts this will be an instance of
        :class:`aa.asset_pricing.univariate.SortConfig` or
        :class:`aa.asset_pricing.double_sort.DoubleSortConfig`.  It is
        copied and mutated for each robustness specification.
    config_list : iterable of RobustnessConfig
        Each element defines a single robustness run with its own
        weighting scheme, breakpoint choice, winsorisation level and
        signal lag.
    config_param : str, default 'config'
        Name of the keyword argument used by ``analysis_fn`` to receive
        the configuration object.  For instance, :func:`aa.asset_pricing.univariate.univariate_sort`
        expects ``config``.
    metric_fn : callable or None, optional
        A function applied to the full analysis result to extract a
        scalar metric (e.g. a high–low spread or a regression slope).  If
        ``None`` (the default), the raw result of ``analysis_fn`` is
        stored in the returned table.
    date_col : str, default 'date'
        Name of the date column used when lagging signals.
    permno_col : str, default 'permno'
        Name of the security identifier column used when lagging signals.
    **data_kwargs : DataFrame or other
        Additional keyword arguments forwarded to ``analysis_fn``.  Any
        argument whose value is a pandas DataFrame and which contains a
        column named ``'signal'`` will be winsorised and lagged according
        to each robustness configuration.  DataFrames without a
        ``'signal'`` column are passed through unchanged.

    Returns
    -------
    DataFrame
        A table where each row corresponds to a robustness
        configuration.  The columns include the fields of
        :class:`RobustnessConfig` plus an additional ``metric`` column
        containing either the scalar extracted by ``metric_fn`` or the
        full analysis result.

    Notes
    -----
    * To switch between equal‑ and value‑weighted returns, the presence
      of the ``size`` DataFrame in ``data_kwargs`` is toggled based on
      the ``weighting`` attribute.  For equal‑weighted runs the ``size``
      keyword argument is replaced with ``None``.
    * Winsorisation applies only to columns explicitly named ``'signal'``.
      Other numeric variables (e.g. returns) are unaffected.  Quantile
      thresholds are computed within the entire sample, not by month or
      cross‑section.
    * Signal lagging is performed within each security (``permno``).  If
      the lag exceeds the available history for a given security, those
      observations are dropped.
    """
    rows: List[dict[str, Any]] = []
    for cfg in config_list:
        # Prepare config for analysis function
        # Use dataclasses.replace to produce a modified copy of base_config
        try:
            new_cfg = replace(
                base_config,
                nyse_breaks=cfg.nyse_breaks,
            )
        except Exception:
            # base_config may not be a dataclass; set attribute manually
            new_cfg = base_config
            if hasattr(new_cfg, "nyse_breaks"):
                object.__setattr__(new_cfg, "nyse_breaks", cfg.nyse_breaks)
        # Build data kwargs for this run
        kwargs_for_run: dict[str, Any] = {}
        for key, val in data_kwargs.items():
            if key == "size":
                # Drop size for equal‑weighted analyses
                kwargs_for_run[key] = val if cfg.weighting.upper() == "VW" else None
                continue
            if isinstance(val, pd.DataFrame) and "signal" in val.columns:
                df = val.copy()
                # Winsorise if requested
                if cfg.winsorize is not None:
                    if isinstance(cfg.winsorize, tuple):
                        limits = cfg.winsorize
                    else:
                        limits = (float(cfg.winsorize), 1.0 - float(cfg.winsorize))
                    df["signal"] = _winsorize_series(df["signal"], limits)
                # Lag if requested
                if cfg.lag:
                    df = _apply_lag(df, cfg.lag, date_col, permno_col)
                kwargs_for_run[key] = df
            else:
                kwargs_for_run[key] = val
        # Set the configuration for the analysis
        kwargs_for_run[config_param] = new_cfg
        # Execute analysis
        try:
            res = analysis_fn(**kwargs_for_run)
        except Exception as exc:
            raise RuntimeError(f"analysis_fn failed for robustness config {cfg}: {exc}")
        # Extract metric if requested
        metric = metric_fn(res) if metric_fn is not None else res
        # Record results
        rows.append(
            {
                "weighting": cfg.weighting,
                "nyse_breaks": cfg.nyse_breaks,
                "winsorize": cfg.winsorize,
                "lag": cfg.lag,
                "holding_period": cfg.holding_period,
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)

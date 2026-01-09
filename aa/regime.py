"""
Regime‑based analysis utilities.

This module provides a simple infrastructure for running any asset‑pricing
analysis separately across different subsamples or market regimes.  In
empirical finance it is common to evaluate whether the performance of an
anomaly is concentrated in certain periods (e.g. before and after 1963,
in bull versus bear markets, or during expansions vs recessions).  The
function :func:`run_by_regime` accepts arbitrary analysis functions
from the rest of the library (such as portfolio sorts or Fama–MacBeth
regressions) and applies them to subsets of the input data defined by
an externally supplied regime indicator.

The design emphasises composability: no hard‑coded notions of what a
"regime" means are built in.  Users supply a `regime_indicator` that
identifies each time period with a label (e.g. ``"high_vol"`` or
``"low_vol"``) and the routine will automatically split the data and
invoke the analysis for each regime.  This makes it trivial to
evaluate robustness across time subsamples, rolling windows or
macroeconomic states.

Examples
--------
>>> from aa.asset_pricing.univariate import univariate_sort, SortConfig
>>> # Create a simple returns and signal DataFrame
>>> returns = pd.DataFrame({"date": pd.date_range("2020-01-31", periods=6, freq="M").repeat(2),
...                        "permno": [1, 2] * 6,
...                        "ret": [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, 0.05, -0.03, 0.02, 0.03]})
>>> signal = returns[["date", "permno"]].copy()
>>> signal["signal"] = [0.5, -0.5] * 6
>>> # Define a regime indicator: first three months vs last three
>>> regime_indicator = pd.DataFrame({"date": pd.date_range("2020-01-31", periods=6, freq="M"),
...                                  "regime": ["early"] * 3 + ["late"] * 3})
>>> res = run_by_regime(
...     regime_indicator=regime_indicator,
...     analysis_fn=univariate_sort,
...     returns=returns,
...     signal=signal,
...     config=SortConfig(n_bins=2)
... )
>>> list(res.keys())
['early', 'late']

Notes
-----
* The `regime_indicator` must have a ``date`` column that aligns with the
  ``date`` column in each DataFrame passed to the analysis function.  The
  module does not assume any specific frequency (monthly, quarterly,
  etc.) but simply matches on exact dates.
* If a given regime has no observations in the relevant DataFrames
  (e.g. because the indicator labels a period without data), the
  corresponding analysis is skipped and an empty dictionary entry is
  returned.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import pandas as pd

__all__ = ["run_by_regime"]


def _prepare_regime_indicator(regime_indicator: Any, date_col: str) -> pd.DataFrame:
    """Validate and coerce a regime indicator into a DataFrame with columns
    ``date`` and ``regime``.

    Parameters
    ----------
    regime_indicator : object
        The regime indicator may be supplied as a pandas Series, a
        DataFrame with a ``date`` column, or a mapping from dates to
        regime labels.  If a Series is provided it must have a datetime
        index and the name of the Series will be ignored.  If a
        dictionary is provided the keys must be datetime‑like and the
        values labels.
    date_col : str
        Name of the date column in the analysis DataFrames.  Used to
        align the indicator.

    Returns
    -------
    DataFrame
        A two‑column DataFrame with columns ``date`` and ``regime``.
    """
    if isinstance(regime_indicator, pd.DataFrame):
        if (
            "date" not in regime_indicator.columns
            or "regime" not in regime_indicator.columns
        ):
            raise KeyError(
                "regime_indicator DataFrame must have 'date' and 'regime' columns"
            )
        df = regime_indicator[["date", "regime"]].copy()
    elif isinstance(regime_indicator, pd.Series):
        if regime_indicator.index.name is None:
            raise ValueError("regime_indicator Series must have a datetime index")
        df = regime_indicator.to_frame(name="regime").reset_index()
        df.rename(columns={regime_indicator.index.name: "date"}, inplace=True)
    elif isinstance(regime_indicator, Mapping):
        # Treat as mapping from dates to labels
        df = pd.DataFrame(list(regime_indicator.items()), columns=["date", "regime"])
    else:
        raise TypeError(
            "regime_indicator must be a DataFrame with 'date' and 'regime' columns, "
            "a Series with datetime index, or a mapping from dates to regime labels"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if df["date"].isna().any():
        raise ValueError("regime_indicator contains invalid dates")
    return df


def run_by_regime(
    *,
    regime_indicator: Any,
    analysis_fn: Callable[..., Any],
    date_col: str = "date",
    **data_kwargs: Any,
) -> Dict[Any, Any]:
    """Apply an analysis function separately to different regimes.

    This routine accepts the same keyword arguments that would be
    supplied to the analysis function and dispatches the call on
    subsets of each DataFrame keyed by the regime indicator.  The
    indicator should specify which dates belong to which regime.

    Parameters
    ----------
    regime_indicator : DataFrame or Series or mapping
        An object encoding the market regimes.  It must resolve to a
        DataFrame with columns ``date`` and ``regime``.  Dates must
        match those in the DataFrames passed through ``data_kwargs``.
    analysis_fn : callable
        A function implementing an empirical analysis (e.g. univariate
        sorts, double sorts, Fama–MacBeth regressions).  It must accept
        the supplied keyword arguments in ``data_kwargs`` and return a
        result for a given subsample.  The function will be called
        repeatedly for each unique regime value.
    date_col : str, default "date"
        Name of the column containing dates in the input DataFrames.
        This column is used to filter the observations for each regime.
    **data_kwargs : DataFrame or any
        Keyword arguments to be forwarded to ``analysis_fn``.  Any
        argument whose value is a pandas DataFrame and which contains
        the ``date_col`` column will be subset by regime; all other
        arguments are passed through unchanged.

    Returns
    -------
    dict
        A dictionary mapping each regime label to the result of
        ``analysis_fn`` on that regime's subset of the data.  If a
        particular regime yields no observations for any of the
        DataFrames, that entry will be omitted.

    Notes
    -----
    The indicator is merged on the date column only.  Therefore, each
    row of an input DataFrame belongs to exactly one regime based on
    its date.  When overlapping frequencies are present (e.g. daily
    returns and monthly regimes), the user should pre‑aggregate or
    interpolate the indicator accordingly.

    Returns empty dictionary for regimes with no data.
    """
    ind_df = _prepare_regime_indicator(regime_indicator, date_col)
    # Collect unique regime labels
    unique_regs = ind_df["regime"].dropna().unique()
    results: Dict[Any, Any] = {}
    # Create a copy of data_kwargs for baseline
    for reg in unique_regs:
        # Determine the set of dates for this regime
        dates = set(ind_df.loc[ind_df["regime"] == reg, "date"])
        if not dates:
            continue
        # Build new keyword args for this regime
        kwargs_for_reg = {}
        skip_regime = True
        for key, val in data_kwargs.items():
            if isinstance(val, pd.DataFrame) and date_col in val.columns:
                # Filter rows whose date is in the regime's dates
                sub = val[val[date_col].isin(dates)].copy()
                kwargs_for_reg[key] = sub
                if not sub.empty:
                    skip_regime = False
            else:
                kwargs_for_reg[key] = val
        if skip_regime:
            # No data for this regime in any DataFrame – skip
            continue
        try:
            results[reg] = analysis_fn(**kwargs_for_reg)
        except Exception as exc:
            # Propagate exceptions with context
            raise RuntimeError(f"analysis_fn failed for regime {reg!r}: {exc}")
    return results

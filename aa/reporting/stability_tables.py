"""
Reporting utilities for stability and robustness analyses.

This submodule extends :mod:`aa.reporting.tables` with functions that
summarise the results of subsample, regime and robustness exercises.
Empirical researchers often wish to present side‑by‑side comparisons
across regimes, specifications or placebo experiments.  The helper
functions here transform dictionaries and DataFrames produced by
:mod:`aa.regime`, :mod:`aa.robustness` and :mod:`aa.simulation`
into tidy tables suitable for inclusion in Markdown or LaTeX
reports.

Functions
---------
stability_table(results_by_regime, metric_fn)
    Construct a table of metrics across regimes.

robustness_table(robust_df)
    Summarise robustness outcomes across specifications.

null_distribution_summary(metrics, observed)
    Provide summary statistics and p‑value for a null distribution.

These functions return DataFrames; use :func:`aa.reporting.tables._format_table`
to convert them into Markdown or LaTeX strings.

"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable

import numpy as np
import pandas as pd

from .tables import _format_table

__all__ = [
    "stability_table",
    "robustness_table",
    "null_distribution_summary",
]


def stability_table(
    results_by_regime: Dict[Any, Any],
    metric_fn: Callable[[Any], Any],
) -> Dict[str, str]:
    """Format a table of metrics across regimes.

    Parameters
    ----------
    results_by_regime : dict
        Mapping from regime label to the full analysis output for that
        regime.
    metric_fn : callable
        Function extracting a scalar or small object from each result.
        For example, if the result is a dictionary from
        :func:`aa.asset_pricing.univariate.univariate_sort`, one might
        extract the high–low equal‑weighted return.

    Returns
    -------
    dict
        Dictionary with keys ``'markdown'`` and ``'latex'`` containing
        formatted tables of the metric by regime.
    """
    rows = []
    for reg, res in results_by_regime.items():
        val = metric_fn(res)
        rows.append({"regime": reg, "metric": val})
    df = pd.DataFrame(rows).set_index("regime")
    return _format_table(df)


def robustness_table(robust_df: pd.DataFrame) -> Dict[str, str]:
    """Summarise robustness outcomes across configurations.

    The returned table includes one row per robustness configuration
    with columns describing the specification and the metric.  If the
    ``metric`` column itself contains scalars (floats), the table
    displays their values directly.  For complex objects, the user
    should pre‑process the DataFrame accordingly.

    Parameters
    ----------
    robust_df : DataFrame
        Output of :func:`aa.robustness.run_robustness_checks`.  Must
        include the columns ``weighting``, ``nyse_breaks``, ``winsorize``,
        ``lag``, ``holding_period`` and ``metric``.

    Returns
    -------
    dict
        ``'markdown'`` and ``'latex'`` representation of the table.
    """
    # For readability convert tuple winsorisation to string
    df = robust_df.copy()
    if "winsorize" in df.columns:
        df["winsorize"] = df["winsorize"].apply(lambda x: "" if x is None else str(x))
    return _format_table(df.set_index(df.columns[:-1].tolist()))


def null_distribution_summary(
    metrics: Iterable[float],
    observed: float,
) -> Dict[str, str]:
    """Summarise a null distribution and compute a two‑sided p‑value.

    Parameters
    ----------
    metrics : iterable of float
        Values of the test statistic under the null (e.g. from
        :func:`aa.simulation.permutation_test` or
        :func:`aa.simulation.simulate_null_distribution`).
    observed : float
        Value of the statistic computed from the actual data.

    Returns
    -------
    dict
        Formatted summary table with columns ``observed``, ``mean_null``,
        ``std_null`` and ``p_value``.  The p‑value is computed as the
        fraction of null statistics whose absolute value exceeds
        ``|observed|`` (two‑sided test).
    """
    arr = np.asarray(list(metrics), dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else float("nan")
    std = float(np.nanstd(arr, ddof=0)) if arr.size else float("nan")
    if arr.size:
        count = np.sum(np.abs(arr) >= abs(observed))
        pval = count / arr.size
    else:
        pval = float("nan")
    summary_df = pd.DataFrame(
        {
            "observed": [observed],
            "mean_null": [mean],
            "std_null": [std],
            "p_value": [pval],
        },
        index=["statistic"],
    )
    return _format_table(summary_df)

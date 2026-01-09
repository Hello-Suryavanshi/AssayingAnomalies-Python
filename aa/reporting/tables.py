"""
Table formatting for asset pricing results.

This module provides helper functions to convert the outputs of
portfolio sorts and Fama–MacBeth regressions into Markdown and
LaTeX tables suitable for academic papers.  The emphasis is on
producing clean tables without external styling dependencies.

The functions return a dictionary with two keys: ``markdown`` and
``latex``.  The value associated with ``markdown`` is a string
containing a GitHub‑flavoured Markdown table, while ``latex``
contains LaTeX code compatible with the ``tabular`` environment.

Usage
-----
>>> from aa.reporting.tables import portfolio_returns_table
>>> res = double_sort(...)
>>> tables = portfolio_returns_table(res['summary'], value_weighted=False)
>>> print(tables['markdown'])
>>> print(tables['latex'])

Notes
-----
These functions do not attempt to adjust units or formatting beyond
rounding floats to three decimal places.  Users seeking greater
control over table appearance should post‑process the returned strings.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

__all__ = ["portfolio_returns_table", "high_low_table", "fama_macbeth_table"]


def _format_table(df: pd.DataFrame) -> Dict[str, str]:
    """Internal helper to format a DataFrame into Markdown and LaTeX.

    Floats are rounded to three decimal places.  The index and column
    names are included in the output.
    """
    # Round numeric columns
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    # Markdown
    md = fmt_df.to_markdown(index=True)
    # LaTeX
    latex = fmt_df.to_latex(index=True, escape=False)
    return {"markdown": md, "latex": latex}


def portfolio_returns_table(
    summary: pd.DataFrame,
    *,
    value_weighted: bool = False,
) -> Dict[str, str]:
    """
    Generate a 2D portfolio return table from a double sort summary.

    Parameters
    ----------
    summary : DataFrame
        Output of :func:`aa.asset_pricing.double_sort` keyed by
        ``'summary'``.  Must contain columns ``bin1``, ``bin2`` and
        either ``ret_ew`` or ``ret_vw`` depending on the desired
        weighting.
    value_weighted : bool, default False
        If True, use the value‑weighted returns (``ret_vw``); if
        False, use equal‑weighted returns (``ret_ew``).

    Returns
    -------
    dict
        Dictionary with keys ``markdown`` and ``latex`` containing
        formatted tables.  The index corresponds to ``bin1`` and
        columns correspond to ``bin2``.

    Notes
    -----
    This function pivots the summary DataFrame so that bins along the
    first characteristic form the rows and bins along the second
    characteristic form the columns.  Any missing combinations are
    represented by ``NaN``.
    """
    col = "ret_vw" if value_weighted else "ret_ew"
    if col not in summary.columns:
        raise KeyError(f"summary must contain column '{col}'")
    pivot = summary.pivot(index="bin1", columns="bin2", values=col)
    # Ensure sorted index and columns
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    # Assign proper names for nicer output
    pivot.index.name = "bin1"
    pivot.columns.name = "bin2"
    return _format_table(pivot)


def high_low_table(
    hl_ts: pd.DataFrame,
    *,
    value_weighted: bool = False,
    average: bool = True,
) -> Dict[str, str]:
    """
    Create a summary table for high–low spreads.

    Parameters
    ----------
    hl_ts : DataFrame
        Time‑series of high–low spreads returned by
        :func:`aa.asset_pricing.double_sort` under keys ``'hl_dim1'``
        or ``'hl_dim2'``.  Must contain columns ``hl_ew`` and
        ``hl_vw``.
    value_weighted : bool, default False
        If True, use ``hl_vw``; otherwise use ``hl_ew``.
    average : bool, default True
        If True, compute the sample mean of the high–low series and
        display as a one‑row table.  If False, return the entire
        time‑series as a table.

    Returns
    -------
    dict
        Dictionary with keys ``markdown`` and ``latex``.
    """
    col = "hl_vw" if value_weighted else "hl_ew"
    if col not in hl_ts.columns:
        raise KeyError(f"hl_ts must contain column '{col}'")
    if average:
        avg = hl_ts[col].mean()
        df = pd.DataFrame({col: [avg]}, index=["mean"])
    else:
        df = hl_ts[["date", col]].copy()
        df = df.set_index("date")
    return _format_table(df)


def fama_macbeth_table(
    fm_results: Dict[str, pd.DataFrame | pd.Series],
) -> Dict[str, str]:
    """
    Format Fama–MacBeth regression results into a table.

    Parameters
    ----------
    fm_results : dict
        Output of :func:`aa.asset_pricing.fama_macbeth_full`.  Must
        include keys ``lambdas``, ``se``, ``tstat`` and ``n_obs``.

    Returns
    -------
    dict
        Formatted table containing the average coefficients
        (``lambda``), Newey–West standard errors (``se``), t‑statistics
        (``t``) and the number of observation periods (``n_obs``).

    Notes
    -----
    Coefficients and standard errors are rounded to three decimal
    places.  t‑statistics are computed externally and therefore
    passed through without additional rounding here.
    """
    lambdas = fm_results.get("lambdas")
    se = fm_results.get("se")
    tstat = fm_results.get("tstat")
    n_obs = fm_results.get("n_obs")
    if not all(obj is not None for obj in (lambdas, se, tstat, n_obs)):
        raise ValueError(
            "fm_results must include 'lambdas', 'se', 'tstat' and 'n_obs' entries"
        )
    df = pd.DataFrame(
        {
            "lambda": lambdas,
            "se": se,
            "t": tstat,
            "n_obs": n_obs,
        }
    )
    # Order rows by index
    df = df.reindex(df.index.tolist())
    return _format_table(df)

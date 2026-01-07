"""
ID normalisation helpers.

These functions provide lightweight utilities for working with CRSP
identifiers.  They are intentionally minimal and documented with
type annotations to support static analysis.
"""

from __future__ import annotations

import pandas as pd


def normalize_permno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the ``permno`` column of a DataFrame to pandas' nullable integer.

    Parameters
    ----------
    df : DataFrame
        A table that may contain a column named ``permno``.

    Returns
    -------
    DataFrame
        The input DataFrame with ``permno`` converted to :class:`pd.Int64Dtype`
        where present.
    """
    out = df.copy()
    if "permno" in out:
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out


def keep_common_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a CRSP monthly table to retain only common equity securities.

    The Assaying Anomalies protocol typically restricts the investment
    universe to CRSP share codes of 10 or 11 (common shares) and may
    further restrict by exchange code.  This helper currently applies
    only the share code filter and can be extended in :mod:`aa.prep`.

    Parameters
    ----------
    df : DataFrame
        Must contain a ``shrcd`` column.

    Returns
    -------
    DataFrame
        Subset of the input with ``shrcd`` equal to 10 or 11.
    """
    if "shrcd" not in df.columns:
        raise KeyError("DataFrame must contain 'shrcd' to filter common equity.")
    return df[df["shrcd"].isin([10, 11])]

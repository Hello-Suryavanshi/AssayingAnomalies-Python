"""
Data loaders for the Assaying Anomalies pipeline.

These functions provide a thin abstraction layer over reading raw
inputs from disk.  In practice, users may wish to supply their own
DataFrames directly (for example in unit tests), and each loader
accepts a fallback DataFrame via the ``df`` argument.

Only Parquet files are supported for on‑disk loading to avoid
introducing heavy dependencies beyond ``pandas``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _read_parquet(path: Path) -> pd.DataFrame:
    """Internal helper to read a Parquet file via pandas."""
    return pd.read_parquet(path)


def load_crsp(
    *, path: Optional[Union[str, Path]] = None, df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Load CRSP monthly data.

    Parameters
    ----------
    path : str or Path, optional
        Location of a Parquet file containing CRSP data.  Ignored if
        ``df`` is provided.
    df : DataFrame, optional
        Provide CRSP data directly instead of reading from disk.

    Returns
    -------
    DataFrame
        The loaded CRSP table.

    Raises
    ------
    FileNotFoundError
        If neither ``df`` nor ``path`` is given.
    """
    if df is not None:
        return df.copy()
    if path is None:
        raise FileNotFoundError(
            "Either a CRSP DataFrame or file path must be provided."
        )
    return _read_parquet(Path(path))


def load_compustat(
    *, path: Optional[Union[str, Path]] = None, df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Load Compustat annual fundamentals (FUNDA).

    Parameters
    ----------
    path : str or Path, optional
        Location of a Parquet file containing FUNDA.  Ignored if
        ``df`` is provided.
    df : DataFrame, optional
        Provide FUNDA data directly.

    Returns
    -------
    DataFrame
        The loaded FUNDA table.
    """
    if df is not None:
        return df.copy()
    if path is None:
        raise FileNotFoundError(
            "Either a FUNDA DataFrame or file path must be provided."
        )
    return _read_parquet(Path(path))


def load_link(
    *, path: Optional[Union[str, Path]] = None, df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Load CCM link history.

    Parameters
    ----------
    path : str or Path, optional
        Location of a Parquet file containing the link history.  Ignored if
        ``df`` is provided.
    df : DataFrame, optional
        Provide link history directly.

    Returns
    -------
    DataFrame
        The loaded link history table.
    """
    if df is not None:
        return df.copy()
    if path is None:
        raise FileNotFoundError(
            "Either a link history DataFrame or file path must be provided."
        )
    return _read_parquet(Path(path))

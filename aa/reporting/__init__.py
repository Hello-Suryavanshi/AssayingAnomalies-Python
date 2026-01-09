"""Reporting utilities for generating publication‑quality tables.

This subpackage contains simple functions to transform the outputs of
sorting and regression routines into Markdown and LaTeX tables.  The
design emphasises minimal dependencies and journal‑style formatting.

Example
-------
>>> from aa.reporting.tables import portfolio_returns_table
>>> table = portfolio_returns_table(summary_df, value_weighted=False)
>>> print(table["markdown"])

"""

from .tables import (
    portfolio_returns_table,
    high_low_table,
    fama_macbeth_table,
)

__all__ = [
    "portfolio_returns_table",
    "high_low_table",
    "fama_macbeth_table",
]

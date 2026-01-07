"""
Formatting helpers for producing summary tables in Markdown.

We intentionally avoid pandas.DataFrame.to_markdown() because it depends on the
optional 'tabulate' package. This module implements a tiny Markdown renderer
using only stdlib + pandas.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def _fmt(x: object, ndp: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if pd.isna(x):
            return ""
        return f"{x:.{ndp}f}"
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime("%Y-%m-%d")
    return str(x)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    # Header
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # Rows
    for r in rows:
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(out)


def portfolio_summary_md(summary: pd.DataFrame) -> str:
    """
    Convert a portfolio sort summary DataFrame to Markdown without tabulate.
    Expected columns: ['bin', 'ret_ew', 'ret_vw'].
    """
    if summary.empty:
        return ""

    cols = ["bin", "ret_ew", "ret_vw"]
    for c in cols:
        if c not in summary.columns:
            raise KeyError(f"summary is missing required column: {c}")

    rows = summary[cols].itertuples(index=False, name=None)
    return _markdown_table(cols, rows)


def regression_table(lambdas: pd.Series, se: pd.Series) -> pd.DataFrame:
    """
    Construct a regression coefficient table.
    """
    df = pd.DataFrame({"coef": lambdas, "se": se})
    df["t"] = df["coef"] / df["se"]
    return df

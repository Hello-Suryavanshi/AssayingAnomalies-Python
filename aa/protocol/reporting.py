from __future__ import annotations

from typing import Any, Mapping, Union
import pandas as pd

PandasObj = Union[pd.Series, pd.DataFrame]


def _to_markdown(x: PandasObj) -> str:
    if isinstance(x, pd.Series):
        return x.to_frame().to_markdown()
    return x.to_markdown()


def to_markdown_tables(res: Mapping[str, Any]) -> str:
    md: list[str] = []

    fmb = res.get("fmb")
    if isinstance(fmb, (pd.Series, pd.DataFrame)):
        md.append("## Fama–MacBeth risk prices\n")
        md.append(_to_markdown(fmb))

    hl = res.get("hl")
    if isinstance(hl, (pd.Series, pd.DataFrame)):
        md.append("\n## H-L monthly series (first 12 rows)\n")
        head = hl.head(12)
        md.append(_to_markdown(head))

    return "\n\n".join(md)

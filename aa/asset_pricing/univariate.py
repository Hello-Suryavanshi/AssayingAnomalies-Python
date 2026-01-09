"""
Univariate portfolio sorts and high‑low (H‑L) series.

This module implements univariate sorts akin to the MATLAB functions
``makeUnivSortInd`` and ``runUnivSort`` found in the Assaying
Anomalies toolkit.  Given asset returns and a cross‑sectional
signal, assets are ranked into ``n_bins`` portfolios each period
using either the full universe or only NYSE stocks to determine
breakpoints.  The module supports both equal‑weighted (EW) and
value‑weighted (VW) returns and produces time‑series and summary
tables along with a high‑minus‑low series.

MATLAB → Python mapping
------------------------
The MATLAB pipeline computes portfolio assignments via
``makeUnivSortInd.m`` and then calls ``runUnivSort.m`` to compute
portfolio returns and long–short spreads.  The Python function
:func:`univariate_sort` below consolidates these steps: it assigns
bins within each period and computes EW and VW returns directly.  The
:class:`SortConfig` dataclass mirrors the optional arguments in the
MATLAB functions (e.g. number of bins, NYSE breakpoints, minimum
observations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["SortConfig", "univariate_sort"]

LS_LABEL = "L\u2011S"  # "L-S" with non-breaking hyphen (U+2011)


@dataclass(frozen=True)
class SortConfig:
    """
    Configuration for univariate sorts.

    Parameters
    ----------
    n_bins : int, default 5
        Number of portfolios to form each period (e.g. 10 deciles, 5 quintiles).
    nyse_breaks : bool, default False
        If True, breakpoints are computed using only NYSE stocks (exchcd == 1).
        If False, the entire universe is used.
    min_obs : int, default 20
        Minimum number of observations required in both the breakpoint universe
        and the full universe for a period to be sorted.
    """

    n_bins: int = 5
    nyse_breaks: bool = False
    min_obs: int = 20


def _bin_edges(x: pd.Series, n: int) -> np.ndarray:
    """Compute robust bin edges for an array of signals."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return np.array([], dtype=float)
    # Prefer quantile-based bins (like MATLAB ranks -> quantiles)
    try:
        _, bins = pd.qcut(x, q=n, retbins=True, duplicates="drop")
        bins = np.asarray(bins, dtype=float)
        if bins.size >= 2 and np.all(np.diff(bins) > 0):
            return bins
    except Exception:
        pass
    # Fallback: equal-width bins
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.array([], dtype=float)
    return np.linspace(xmin, xmax, num=n + 1, dtype=float)


def univariate_sort(
    *,
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    size: Optional[pd.DataFrame] = None,
    exch: Optional[pd.DataFrame] = None,
    config: SortConfig = SortConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Perform univariate portfolio sorts and return time‑series and summary.

    Parameters
    ----------
    returns : DataFrame
        Columns: date, permno, ret
    signal : DataFrame
        Columns: date, permno, signal
    size : DataFrame, optional
        Columns: date, permno, me (market equity used for VW weights)
    exch : DataFrame, optional
        Columns: date, permno, exchcd (NYSE == 1)
    config : SortConfig, optional
        Sort settings (bins, NYSE breakpoints, min_obs)

    Returns
    -------
    dict with keys:
      - ``time_series``: DataFrame with columns date, bin, ret_ew, ret_vw
      - ``summary``: DataFrame with mean ret_ew/ret_vw by bin and one L‑S row
    """
    # Normalize dates
    for df in (returns, signal, size, exch):
        if df is not None and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(
                None
            )
    # Merge inputs
    base = signal.merge(returns, on=["date", "permno"], how="inner", validate="m:1")
    if size is not None:
        base = base.merge(
            size[["date", "permno", "me"]], on=["date", "permno"], how="left"
        )
    if exch is not None:
        base = base.merge(
            exch[["date", "permno", "exchcd"]], on=["date", "permno"], how="left"
        )
    # Ensure required cols exist
    if "signal" not in base.columns or "ret" not in base.columns:
        raise KeyError("Merged data must contain 'signal' and 'ret' columns.")
    # If exchcd missing, treat as non-NYSE
    if "exchcd" not in base.columns:
        base["exchcd"] = np.nan

    def month_sort(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        bp_univ = g[g["exchcd"] == 1] if config.nyse_breaks else g
        if len(bp_univ) < config.min_obs or len(g) < config.min_obs:
            return pd.DataFrame(columns=["date", "bin", "ret_ew", "ret_vw"])
        edges = _bin_edges(bp_univ["signal"], config.n_bins)
        if edges.size < 2:
            return pd.DataFrame(columns=["date", "bin", "ret_ew", "ret_vw"])
        bins_list: list[float] = [
            float(v) for v in np.asarray(edges, dtype=float).tolist()
        ]
        g["bin"] = pd.cut(
            g["signal"], bins=bins_list, labels=False, include_lowest=True, right=True
        )
        if g["bin"].isna().all():
            return pd.DataFrame(columns=["date", "bin", "ret_ew", "ret_vw"])
        g["bin"] = (g["bin"].astype("Int64") + 1).astype("Int64")
        # Equal‑weighted
        ew = g.groupby("bin", as_index=False).agg(ret_ew=("ret", "mean"))
        # Value‑weighted (if 'me' present and finite)
        if size is not None and "me" in g.columns:
            tmp = g[["bin", "ret", "me"]].copy()
            tmp = tmp[np.isfinite(tmp["me"].to_numpy(dtype=float))]
            if tmp.empty:
                vw = ew[["bin"]].copy()
                vw["ret_vw"] = np.nan
            else:
                tmp["wret"] = tmp["me"] * tmp["ret"]
                vw_sum = tmp.groupby("bin", as_index=False)[["wret", "me"]].sum()
                vw_sum["ret_vw"] = vw_sum["wret"] / vw_sum["me"]
                vw = vw_sum[["bin", "ret_vw"]]
        else:
            vw = ew[["bin"]].copy()
            vw["ret_vw"] = np.nan
        out = ew.merge(vw, on="bin", how="left")
        out["date"] = g["date"].iloc[0]
        out["bin"] = out["bin"].astype("Int64")
        return out[["date", "bin", "ret_ew", "ret_vw"]]

    # Build time‑series
    pieces: list[pd.DataFrame] = []
    for dt, g in base.groupby("date", sort=True):
        res = month_sort(g)
        if not res.empty:
            pieces.append(res)
    if pieces:
        ts = pd.concat(pieces, ignore_index=True)
    else:
        ts = pd.DataFrame(columns=["date", "bin", "ret_ew", "ret_vw"])
    # Summary across time
    summ = ts.groupby("bin", as_index=False)[["ret_ew", "ret_vw"]].mean()
    # Add L-S row
    if (
        not summ.empty
        and pd.api.types.is_integer_dtype(summ["bin"])
        and summ["bin"].max() >= 2
    ):
        k = int(summ["bin"].max())

        def safe_item(frame: pd.DataFrame, b: int, col: str) -> float:
            s = frame.loc[frame["bin"] == b, col]
            return float(s.iloc[0]) if len(s) else float(np.nan)

        ls = pd.DataFrame(
            {
                "bin": [LS_LABEL],
                "ret_ew": [safe_item(summ, k, "ret_ew") - safe_item(summ, 1, "ret_ew")],
                "ret_vw": [safe_item(summ, k, "ret_vw") - safe_item(summ, 1, "ret_vw")],
            }
        )
        summ = pd.concat([summ, ls], ignore_index=True)
    return {"time_series": ts, "summary": summ}

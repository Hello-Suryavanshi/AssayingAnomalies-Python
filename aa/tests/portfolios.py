"""
Minimal univariate portfolio sorts (EW & VW), optional NYSE breakpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, cast

import numpy as np
import pandas as pd

__all__ = ["SortConfig", "univariate_sort"]


@dataclass(frozen=True)
class SortConfig:
    n_bins: int = 5
    nyse_breaks: bool = False
    min_obs: int = 20


def _bin_edges(x: pd.Series, n: int) -> np.ndarray:
    """
    Robust bin edge finder:
    - Uses qcut with duplicates='drop' to ensure strictly increasing edges.
    - Falls back to linear-spaced edges over data range if qcut fails.
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return np.array([], dtype=float)

    try:
        _, bins = pd.qcut(x, q=n, retbins=True, duplicates="drop")
        bins = np.asarray(bins, dtype=float)
        if bins.size >= 2 and np.all(np.diff(bins) > 0):
            return bins
    except Exception:
        pass

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
    Build univariate sorts each month.

    Expected columns:
      - signal: ['date','permno','signal']
      - returns: ['date','permno','ret']
      - size (optional): ['date','permno','me']  (for VW)
      - exch (optional): ['date','permno','exchcd']  (NYSE==1 for breakpoints)

    Returns:
      dict(time_series=DataFrame, summary=DataFrame)
    """

    # Normalize dates to naive Timestamps
    for df in (returns, signal, size, exch):
        if df is not None and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(
                None
            )

    base = signal.merge(returns, on=["date", "permno"], how="inner", validate="m:1")
    if size is not None:
        base = base.merge(
            size[["date", "permno", "me"]], on=["date", "permno"], how="left"
        )
    if exch is not None:
        base = base.merge(
            exch[["date", "permno", "exchcd"]], on=["date", "permno"], how="left"
        )

    def _month_sort(g: pd.DataFrame) -> pd.DataFrame:
        # Breakpoint universe
        if config.nyse_breaks and "exchcd" in g.columns:
            bp_univ = g[g["exchcd"] == 1]
        else:
            bp_univ = g

        # Guardrails for tiny months
        if len(bp_univ) < config.min_obs or len(g) < config.min_obs:
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])

        # Bin edges from breakpoint universe
        edges = _bin_edges(bp_univ["signal"], config.n_bins)
        if edges.size < 2:
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])

        # Assign bins (1..k)
        edges_seq: Sequence[float] = cast(Sequence[float], edges.tolist())
        bins = pd.cut(
            g["signal"].astype(float),
            bins=edges_seq,
            labels=False,
            include_lowest=True,
            right=True,
        )
        if bins.isna().all():
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])
        g = g.copy()
        g["bin"] = (bins.astype("Int64") + 1).astype("Int64")

        # Vectorized EW & VW per bin — no GroupBy.apply
        # EW: simple mean of ret
        # VW: sum(me*ret)/sum(me), guarding denom<=0 or missing 'me'
        g["ret"] = pd.to_numeric(g["ret"], errors="coerce")
        if "me" in g.columns:
            g["me"] = pd.to_numeric(g["me"], errors="coerce")
            g["wr"] = g["me"] * g["ret"]
            agg = g.groupby("bin", as_index=False).agg(
                ret_ew=("ret", "mean"), sum_me=("me", "sum"), sum_wr=("wr", "sum")
            )
            # Safe VW
            with np.errstate(invalid="ignore", divide="ignore"):
                ret_vw = agg["sum_wr"] / agg["sum_me"]
            agg = agg.drop(columns=["sum_me", "sum_wr"])
            agg["ret_vw"] = ret_vw.replace([np.inf, -np.inf], np.nan)
        else:
            agg = g.groupby("bin", as_index=False).agg(ret_ew=("ret", "mean"))
            agg["ret_vw"] = np.nan

        agg["bin"] = agg["bin"].astype("Int64")
        return agg[["bin", "ret_ew", "ret_vw"]]

    # Monthly loop (avoid GroupBy.apply for clean typing)
    frames: list[pd.DataFrame] = []
    for d, g in base.groupby("date", sort=False):
        out = _month_sort(g)
        if not out.empty:
            out = out.copy()
            out.insert(0, "date", pd.Timestamp(d))
            frames.append(out)

    if frames:
        ts = pd.concat(frames, ignore_index=True)
    else:
        ts = pd.DataFrame(columns=["date", "bin", "ret_ew", "ret_vw"])

    # Cleanups
    ts = ts.dropna(subset=["ret_ew", "ret_vw"], how="all")

    # Summary: mean by bin + L-S
    if ts.empty:
        summ = pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])
    else:
        summ = ts.groupby("bin", as_index=False)[["ret_ew", "ret_vw"]].mean()

    if not summ.empty and pd.to_numeric(summ["bin"], errors="coerce").max() >= 2:
        k = int(pd.to_numeric(summ["bin"], errors="coerce").max())

        def _safe_item(frame: pd.DataFrame, b: int, col: str) -> float:
            s = frame.loc[frame["bin"] == b, col]
            return float(s.item()) if len(s) == 1 else float("nan")

        ls = pd.DataFrame(
            {
                "bin": pd.Series(["L-S"], dtype="object"),
                "ret_ew": [
                    _safe_item(summ, k, "ret_ew") - _safe_item(summ, 1, "ret_ew")
                ],
                "ret_vw": [
                    _safe_item(summ, k, "ret_vw") - _safe_item(summ, 1, "ret_vw")
                ],
            }
        )
        summ = pd.concat([summ, ls], ignore_index=True)

    return {"time_series": ts, "summary": summ}

"""
Minimal univariate portfolio sorts (EW & VW), optional NYSE breakpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
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
    # Normalize dates
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
        g = g.copy()
        if config.nyse_breaks and "exchcd" in g:
            bp_univ = g[g["exchcd"] == 1]
        else:
            bp_univ = g

        if len(bp_univ) < config.min_obs or len(g) < config.min_obs:
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])

        edges = _bin_edges(bp_univ["signal"], config.n_bins)
        if edges.size < 2:
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])

        # Keep only the overload ignore (mypy); arg-type is no longer needed
        g["bin"] = pd.cut(  # type: ignore[call-overload]
            g["signal"], bins=edges, labels=False, include_lowest=True, right=True
        )
        if g["bin"].isna().all():
            return pd.DataFrame(columns=["bin", "ret_ew", "ret_vw"])
        g["bin"] = (g["bin"].astype("Int64") + 1).astype("Int64")

        def _vw(x: pd.DataFrame) -> float:
            if "me" not in x or not np.isfinite(x["me"]).any():
                return float("nan")
            w = np.where(np.isfinite(x["me"]), x["me"], 0.0)
            denom = float(np.nansum(w))
            if denom <= 0:
                return float("nan")
            return float(np.nansum(w * x["ret"]) / denom)

        out = (
            g.groupby("bin", as_index=False)
            .apply(
                lambda x: pd.Series(
                    {"ret_ew": float(np.nanmean(x["ret"])), "ret_vw": _vw(x)}
                )
            )
            .reset_index(drop=True)
        )
        out["bin"] = out["bin"].astype("Int64")
        return out

    ts = base.groupby("date", as_index=False).apply(_month_sort).reset_index()
    if "level_0" in ts.columns:
        ts = ts.drop(columns=["level_0"])
    ts = ts.dropna(subset=["ret_ew", "ret_vw"], how="all")

    summ = ts.groupby("bin", as_index=False)[["ret_ew", "ret_vw"]].mean()
    if not summ.empty and summ["bin"].max() >= 2:
        k = int(summ["bin"].max())

        def _safe_item(frame: pd.DataFrame, b: int, col: str) -> float:
            s = frame.loc[frame["bin"] == b, col]
            return float(s.item()) if len(s) == 1 else float("nan")

        ls = pd.DataFrame(
            {
                "bin": ["L-S"],
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

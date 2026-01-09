"""
Two-dimensional (double) portfolio sorts.

This module implements double portfolio sorts analogous to the
``runDoubleSort`` routine in the MATLAB Assaying Anomalies code. In
contrast to univariate sorts, double sorts form portfolios on two
characteristics simultaneously. Both independent and conditional
sorts are supported. The resulting panel of portfolio returns can
then be summarised or used to compute high–low spreads along either
dimension.

Given monthly returns and two cross-sectional signals, the function
:func:`double_sort` assigns each stock to a bin along both
characteristics in every month. Equal-weighted and value-weighted
portfolio returns are computed without relying on GroupBy.apply for
performance and type safety.

Examples
--------
>>> from aa.asset_pricing.double_sort import double_sort, DoubleSortConfig
>>> res = double_sort(
...     returns=returns_df,
...     signal_1=size_signal,
...     signal_2=book_to_market_signal,
...     size=size_df,
...     exch=exch_df,
...     config=DoubleSortConfig(n_bins_1=2, n_bins_2=3, conditional=True)
... )
>>> res["summary"].head()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .univariate import _bin_edges


@dataclass(frozen=True)
class DoubleSortConfig:
    """
    Configuration for double portfolio sorts.

    Parameters
    ----------
    n_bins_1 : int, default 3
        Number of portfolios along the first characteristic (rows).
    n_bins_2 : int, default 3
        Number of portfolios along the second characteristic (columns).
    nyse_breaks : bool, default False
        If True, breakpoints are computed using only NYSE stocks
        (``exchcd == 1``) for both characteristics. If False, all
        stocks are used to determine breakpoints.
    min_obs : int, default 20
        Minimum number of observations required in the breakpoint
        universe and the full universe to form portfolios in a given
        month.
    conditional : bool, default False
        If True, perform a conditional sort: first sort on
        ``signal_1``, then within each ``signal_1`` bin compute
        breakpoints for ``signal_2``. If False, the two
        characteristics are sorted independently.
    """

    n_bins_1: int = 3
    n_bins_2: int = 3
    nyse_breaks: bool = False
    min_obs: int = 20
    conditional: bool = False


def _assign_bins(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign integer bins based on provided edges.

    Returns a 1-based integer series; NaN values remain NA.
    """
    if edges.size < 2:
        return pd.Series(index=series.index, dtype="Int64")
    labels = pd.cut(
        series.astype(float),
        bins=[float(v) for v in edges.tolist()],
        labels=False,
        include_lowest=True,
        right=True,
    )
    return (labels.astype("Int64") + 1).astype("Int64")


def double_sort(
    *,
    returns: pd.DataFrame,
    signal_1: pd.DataFrame,
    signal_2: pd.DataFrame,
    size: Optional[pd.DataFrame] = None,
    exch: Optional[pd.DataFrame] = None,
    config: DoubleSortConfig = DoubleSortConfig(),
) -> dict[str, pd.DataFrame]:
    """
    Perform double portfolio sorts on two characteristics.

    Parameters
    ----------
    returns : DataFrame
        Monthly returns with columns ``date``, ``permno`` and ``ret``.
    signal_1 : DataFrame
        First characteristic with columns ``date``, ``permno`` and ``signal``.
    signal_2 : DataFrame
        Second characteristic with columns ``date``, ``permno`` and ``signal``.
    size : DataFrame, optional
        Market equity for value-weighted returns. Columns: ``date``, ``permno``, ``me``.
    exch : DataFrame, optional
        Exchange codes for NYSE breakpoints. Columns: ``date``, ``permno``, ``exchcd``.
    config : DoubleSortConfig, optional
        Sort settings.

    Returns
    -------
    dict
        ``time_series``: DataFrame with columns ``date``, ``bin1``, ``bin2``,
        ``ret_ew`` and ``ret_vw`` containing monthly portfolio returns.
        ``summary``: DataFrame with mean ``ret_ew`` and ``ret_vw`` by (bin1, bin2).
        ``hl_dim1`` and ``hl_dim2``: DataFrames of high–low spreads along the first
        and second characteristic respectively.

    Notes
    -----
    Independent sorts compute breakpoints for each characteristic from the full
    (or NYSE) universe. Conditional sorts first assign ``signal_1`` bins, then
    compute breakpoints for ``signal_2`` separately within each ``signal_1`` bin.
    """
    # Normalise dates
    for df in (returns, signal_1, signal_2, size, exch):
        if df is not None and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(
                None
            )

    # Merge all inputs
    base = signal_1.rename(columns={"signal": "signal1"}).merge(
        signal_2.rename(columns={"signal": "signal2"}),
        on=["date", "permno"],
        how="inner",
        validate="m:1",
    )
    base = base.merge(
        returns[["date", "permno", "ret"]], on=["date", "permno"], how="inner"
    )
    if size is not None:
        base = base.merge(
            size[["date", "permno", "me"]], on=["date", "permno"], how="left"
        )
    if exch is not None:
        base = base.merge(
            exch[["date", "permno", "exchcd"]], on=["date", "permno"], how="left"
        )

    # Ensure exchcd exists
    if "exchcd" not in base.columns:
        base["exchcd"] = np.nan

    ts_frames: list[pd.DataFrame] = []
    hl1_frames: list[pd.DataFrame] = []  # high–low along dim1 per month
    hl2_frames: list[pd.DataFrame] = []  # high–low along dim2 per month

    # Iterate month by month
    for dt, g in base.groupby("date", sort=True):
        g = g.copy()

        # Breakpoint universes
        if config.nyse_breaks:
            bp_univ = g[g["exchcd"] == 1]
        else:
            bp_univ = g

        # Guardrails
        if len(bp_univ) < config.min_obs or len(g) < config.min_obs:
            continue

        # Edges for first signal
        edges1 = _bin_edges(bp_univ["signal1"], config.n_bins_1)
        if edges1.size < 2:
            continue

        g["bin1"] = _assign_bins(g["signal1"], edges1)
        if g["bin1"].isna().all():
            continue

        # Assign second signal bins
        if config.conditional:
            g["bin2"] = pd.NA
            for b1 in g["bin1"].dropna().unique():
                b1_int = int(b1)
                if config.nyse_breaks:
                    sub_univ = bp_univ[bp_univ["bin1"] == b1]
                else:
                    sub_univ = g[g["bin1"] == b1]

                sub_edges = _bin_edges(sub_univ["signal2"], config.n_bins_2)
                if sub_edges.size < 2:
                    g.loc[g["bin1"] == b1, "bin2"] = pd.NA
                else:
                    g.loc[g["bin1"] == b1, "bin2"] = _assign_bins(
                        g.loc[g["bin1"] == b1, "signal2"], sub_edges
                    )
                _ = b1_int  # silence “unused” in type-checkers (no runtime effect)
        else:
            edges2 = _bin_edges(bp_univ["signal2"], config.n_bins_2)
            if edges2.size < 2:
                continue
            g["bin2"] = _assign_bins(g["signal2"], edges2)

        # Drop rows with missing bins
        g = g.dropna(subset=["bin1", "bin2"], how="any")
        if g.empty:
            continue

        # Ensure numeric types
        g["bin1"] = g["bin1"].astype("Int64")
        g["bin2"] = g["bin2"].astype("Int64")
        g["ret"] = pd.to_numeric(g["ret"], errors="coerce")

        # Equal-weighted returns
        ew = g.groupby(["bin1", "bin2"], as_index=False).agg(ret_ew=("ret", "mean"))

        # Value-weighted returns
        if size is not None and "me" in g.columns:
            g["me"] = pd.to_numeric(g["me"], errors="coerce")
            g["wret"] = g["me"] * g["ret"]
            vw_sum = g.groupby(["bin1", "bin2"], as_index=False)[["wret", "me"]].sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                vw_sum["ret_vw"] = vw_sum["wret"] / vw_sum["me"]
            vw = vw_sum[["bin1", "bin2", "ret_vw"]]
        else:
            vw = ew[["bin1", "bin2"]].copy()
            vw["ret_vw"] = np.nan

        out = ew.merge(vw, on=["bin1", "bin2"], how="left")
        out["date"] = dt
        ts_frames.append(out[["date", "bin1", "bin2", "ret_ew", "ret_vw"]])

        # High–low along dimension 1: within each bin2, high bin1 minus low bin1
        if not out.empty:
            b1_min = int(out["bin1"].min())
            b1_max = int(out["bin1"].max())

            hl1_list: list[tuple[int, float, float]] = []
            for b2 in sorted(out["bin2"].dropna().astype(int).unique()):
                sub = out[out["bin2"].astype(int) == b2]
                r_high = sub.loc[sub["bin1"] == b1_max]
                r_low = sub.loc[sub["bin1"] == b1_min]
                if len(r_high) and len(r_low):
                    diff_ew = float(r_high["ret_ew"].iloc[0]) - float(
                        r_low["ret_ew"].iloc[0]
                    )
                    diff_vw = float(r_high["ret_vw"].iloc[0]) - float(
                        r_low["ret_vw"].iloc[0]
                    )
                    hl1_list.append((b2, diff_ew, diff_vw))

            if hl1_list:
                hl1_frames.append(
                    pd.DataFrame(
                        {
                            "date": [dt],
                            "hl_ew": [np.nanmean([x[1] for x in hl1_list])],
                            "hl_vw": [np.nanmean([x[2] for x in hl1_list])],
                        }
                    )
                )

        # High–low along dimension 2: within each bin1, high bin2 minus low bin2
        if not out.empty:
            b2_min = int(out["bin2"].min())
            b2_max = int(out["bin2"].max())

            hl2_list: list[tuple[int, float, float]] = []
            for b1 in sorted(out["bin1"].dropna().astype(int).unique()):
                sub = out[out["bin1"].astype(int) == b1]
                r_high = sub.loc[sub["bin2"] == b2_max]
                r_low = sub.loc[sub["bin2"] == b2_min]
                if len(r_high) and len(r_low):
                    diff_ew = float(r_high["ret_ew"].iloc[0]) - float(
                        r_low["ret_ew"].iloc[0]
                    )
                    diff_vw = float(r_high["ret_vw"].iloc[0]) - float(
                        r_low["ret_vw"].iloc[0]
                    )
                    hl2_list.append((b1, diff_ew, diff_vw))

            if hl2_list:
                hl2_frames.append(
                    pd.DataFrame(
                        {
                            "date": [dt],
                            "hl_ew": [np.nanmean([x[1] for x in hl2_list])],
                            "hl_vw": [np.nanmean([x[2] for x in hl2_list])],
                        }
                    )
                )

    # Concatenate time series
    ts = (
        pd.concat(ts_frames, ignore_index=True)
        if ts_frames
        else pd.DataFrame(columns=["date", "bin1", "bin2", "ret_ew", "ret_vw"])
    )
    hl1_ts = (
        pd.concat(hl1_frames, ignore_index=True)
        if hl1_frames
        else pd.DataFrame(columns=["date", "hl_ew", "hl_vw"])
    )
    hl2_ts = (
        pd.concat(hl2_frames, ignore_index=True)
        if hl2_frames
        else pd.DataFrame(columns=["date", "hl_ew", "hl_vw"])
    )

    # Summary over time
    summary = (
        ts.groupby(["bin1", "bin2"], as_index=False)[["ret_ew", "ret_vw"]].mean()
        if not ts.empty
        else pd.DataFrame(columns=["bin1", "bin2", "ret_ew", "ret_vw"])
    )

    return {
        "time_series": ts,
        "summary": summary,
        "hl_dim1": hl1_ts,
        "hl_dim2": hl2_ts,
    }

# mypy: ignore-errors

"""
Executable pipeline for the size anomaly.

This module wires together the loading, preparation, signal construction,
portfolio sorting, and regression components needed to reproduce the core
logic of ``use_library.m`` from the Assaying Anomalies MATLAB toolkit.
It can be run as a script via ``python -m aa.pipeline.run_size_pipeline``.

The pipeline performs the following steps:

1. Load CRSP monthly data, Compustat FUNDA and CCM link history via
   :mod:`aa.io.loaders`.  Users may specify file paths or supply
   DataFrames directly.
2. Construct a unified monthly panel via :func:`aa.prep.build_monthly_panel`.
3. Compute the lagged size signal (−log ME) using
   :func:`aa.signals.compute_size_signal`.
4. Run univariate portfolio sorts on the signal to obtain equal‑ and
   value‑weighted returns across bins and a high–low series using
   :func:`aa.asset_pricing.univariate_sort`.
5. Estimate Fama–MacBeth cross‑sectional regressions of returns on the
   size signal with Newey–West standard errors using
   :func:`aa.asset_pricing.fama_macbeth`.
6. Produce Markdown‑ready summary tables via :mod:`aa.reporting`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


import pandas as pd

from ..asset_pricing import SortConfig, fama_macbeth, univariate_sort
from ..io import load_compustat, load_crsp, load_link
from ..prep import build_monthly_panel
from ..reporting.tables import portfolio_returns_table, fama_macbeth_table
from ..signals import compute_size_signal

from typing import TypedDict


class SizePipelineResult(TypedDict):
    portfolio_summary: pd.DataFrame
    hl_series: pd.Series | None
    fmb_lambdas: pd.Series
    fmb_se: pd.Series
    fmb_ts: pd.DataFrame


def run_pipeline(
    *,
    crsp: Optional[pd.DataFrame] = None,
    funda: Optional[pd.DataFrame] = None,
    lnkhist: Optional[pd.DataFrame] = None,
    crsp_path: Optional[str | Path] = None,
    funda_path: Optional[str | Path] = None,
    link_path: Optional[str | Path] = None,
    n_bins: int = 5,
    nyse_breaks: bool = True,
    min_obs: int = 50,
) -> SizePipelineResult:
    """
    Execute the size anomaly pipeline and return results.

    Users may supply input data either as DataFrames or as file paths to
    Parquet files.  When both are provided for a dataset, the DataFrame
    takes precedence.

    Parameters
    ----------
    crsp, funda, lnkhist : DataFrame, optional
        Pre‑loaded CRSP, Compustat and CCM data.  At minimum, ``crsp`` must
        have columns ``date``, ``permno``, ``ret``, ``me``, and optionally
        ``exchcd``.  ``funda`` and ``lnkhist`` are optional for the size
        signal but accepted for future extensions.
    crsp_path, funda_path, link_path : str or Path, optional
        File paths to Parquet files containing the respective datasets.
    n_bins : int, default 5
        Number of portfolios for univariate sorts.
    nyse_breaks : bool, default True
        Whether to use NYSE firms to compute breakpoint cut‑offs.
    min_obs : int, default 50
        Minimum number of observations per period required to compute a sort.

    Returns
    -------
    dict
        A dictionary with keys:
        ``"portfolio_summary"`` → DataFrame of mean returns by bin with L‑S row,
        ``"hl_series"`` → Series of time‑series high‑minus‑low returns,
        ``"fmb_lambdas"`` → Series of average Fama–MacBeth coefficients,
        ``"fmb_se"`` → Series of Newey–West standard errors,
        ``"fmb_ts"`` → DataFrame of period‑by‑period coefficient estimates.
    """
    # Load data, prioritising in‑memory objects
    crsp_df = load_crsp(path=crsp_path, df=crsp)
    funda_df = None
    lnkhist_df = None
    if funda is not None or funda_path is not None:
        funda_df = load_compustat(path=funda_path, df=funda)
    if lnkhist is not None or link_path is not None:
        lnkhist_df = load_link(path=link_path, df=lnkhist)

    # Build unified panel (currently a pass‑through when funda/lnkhist are absent)
    panel = build_monthly_panel(crsp=crsp_df, funda=funda_df, lnkhist=lnkhist_df)

    # Construct size signal
    signal = compute_size_signal(panel)

    # Prepare returns, size and exch frames
    returns = panel[["date", "permno", "ret"]].copy()
    size_df = panel[["date", "permno", "me"]].copy()
    exch_df = (
        panel[["date", "permno", "exchcd"]].copy()
        if "exchcd" in panel.columns
        else None
    )
    sort_cfg = SortConfig(n_bins=n_bins, nyse_breaks=nyse_breaks, min_obs=min_obs)
    sort_res = univariate_sort(
        returns=returns,
        signal=signal,
        size=size_df,
        exch=exch_df,
        config=sort_cfg,
    )
    ts = sort_res["time_series"]
    summ = sort_res["summary"]

    # High‑minus‑low series from time‑series output
    hl_series = None
    if not ts.empty and pd.api.types.is_integer_dtype(ts["bin"]):
        k = int(ts["bin"].max())
        # pivot to get returns by bin and compute difference between top and bottom
        pivot = ts.pivot_table(index="date", columns="bin", values="ret_vw")
        if 1 in pivot.columns and k in pivot.columns:
            hl_series = pivot[k] - pivot[1]
            hl_series.name = "hl"

    # Fama–MacBeth regression: need yyyymm integer
    panel_reg = signal.merge(
        returns, on=["date", "permno"], how="inner", validate="m:1"
    ).copy()
    panel_reg["yyyymm"] = panel_reg["date"].dt.year * 100 + panel_reg["date"].dt.month
    fmb_lambdas, fmb_ts, fmb_se = fama_macbeth(
        panel_reg, y="ret", xcols=["signal"], time_col="yyyymm"
    )

    return {
        "portfolio_summary": summ,
        "hl_series": hl_series,
        "fmb_lambdas": fmb_lambdas,
        "fmb_se": fmb_se,
        "fmb_ts": fmb_ts,
    }


def main() -> None:
    """
    Run the size anomaly pipeline and print summary tables.

    This function serves as a convenience entry point when the module is
    executed as a script.  It attempts to load CRSP, Compustat and CCM
    data from common default locations (``data/cache``) and prints the
    resulting tables.  Errors encountered during loading will be raised.
    """
    # Default locations (relative to the current working directory)
    default_dir = Path("data/cache")
    crsp_path = default_dir / "crsp_msf.parquet"
    funda_path = default_dir / "comp_funda.parquet"
    link_path = default_dir / "ccm_lnkhist.parquet"

    res = run_pipeline(
        crsp_path=crsp_path,
        funda_path=funda_path if funda_path.exists() else None,
        link_path=link_path if link_path.exists() else None,
    )

    summ = res["portfolio_summary"]
    hl = res["hl_series"]
    lambdas = res["fmb_lambdas"]
    se = res["fmb_se"]

    print(portfolio_returns_table(summ)["markdown"])
    print(fama_macbeth_table({"lambdas": lambdas, "se": se})["markdown"])

    print("# Portfolio Sorts on Size Signal")
    print("## Summary (mean monthly returns by bin; L-S is top minus bottom):")
    print(portfolio_summary_md(summ))
    if hl is not None:
        print("\n## High‑Minus‑Low Series (VW):")
        hl_df = hl.reset_index().rename(columns={"index": "date"})
        from aa.reporting.tables import _markdown_table

        headers = list(hl_df.columns)
        rows = hl_df.itertuples(index=False, name=None)
        print(_markdown_table(headers, rows))
    print("\n# Fama–MacBeth Regression on Size")
    reg_table = regression_table(lambdas, se)

    from aa.reporting.tables import _markdown_table

    reg_df = reg_table.reset_index().rename(columns={"index": "param"})
    headers = list(reg_df.columns)
    rows = reg_df.itertuples(index=False, name=None)
    print(_markdown_table(headers, rows))


if __name__ == "__main__":
    main()

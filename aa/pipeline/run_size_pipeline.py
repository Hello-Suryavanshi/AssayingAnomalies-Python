# mypy: ignore-errors
"""
Executable pipeline for the size anomaly.

Reproduces core logic of `use_library.m` from the MATLAB Assaying Anomalies toolkit.

Run:
    python -m aa.pipeline.run_size_pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd

from ..asset_pricing import SortConfig, fama_macbeth, univariate_sort
from ..io import load_compustat, load_crsp, load_link
from ..prep import build_monthly_panel
from ..reporting.tables import fama_macbeth_table, portfolio_returns_table
from ..signals import compute_size_signal


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
    """Execute the size anomaly pipeline and return results."""
    crsp_df = load_crsp(path=crsp_path, df=crsp)

    funda_df = (
        load_compustat(path=funda_path, df=funda)
        if (funda is not None or funda_path is not None)
        else None
    )
    lnkhist_df = (
        load_link(path=link_path, df=lnkhist)
        if (lnkhist is not None or link_path is not None)
        else None
    )

    panel = build_monthly_panel(crsp=crsp_df, funda=funda_df, lnkhist=lnkhist_df)

    signal = compute_size_signal(panel)

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

    hl_series = None
    if not ts.empty and {"bin", "ret_vw"}.issubset(ts.columns):
        try:
            k = int(ts["bin"].max())
            pivot = ts.pivot_table(index="date", columns="bin", values="ret_vw")
            if 1 in pivot.columns and k in pivot.columns:
                hl_series = pivot[k] - pivot[1]
                hl_series.name = "hl"
        except Exception:
            hl_series = None

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
    """Run the pipeline from default `data/cache` locations and print Markdown tables."""
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
    lambdas = res["fmb_lambdas"]
    se = res["fmb_se"]

    print("# Portfolio Sorts on Size Signal")
    print(portfolio_returns_table(summ)["markdown"])

    print("\n# Fama–MacBeth Regression on Size")
    fm_out = {
        "lambdas": lambdas,
        "se": se,
        "tstat": lambdas / se,
        "n_obs": len(lambdas),
    }
    print(fama_macbeth_table(fm_out)["markdown"])


if __name__ == "__main__":
    main()

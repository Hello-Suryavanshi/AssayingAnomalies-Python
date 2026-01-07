"""
Integration test for the size anomaly pipeline.

This test constructs a tiny synthetic CRSP panel with Compustat and CCM
tables and verifies that the full end‑to‑end pipeline executes without
error and returns non‑empty, correctly shaped outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aa.pipeline.run_size_pipeline import run_pipeline


def _make_crsp(n_months: int = 6, n_firms: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    permnos = list(range(n_firms))
    idx = pd.MultiIndex.from_product([dates, permnos], names=["date", "permno"])
    df = idx.to_frame(index=False)
    df["me"] = np.exp(rng.normal(8.0, 0.3, size=len(df)))
    signal = -np.log(df["me"])
    df["ret"] = (
        0.02
        + 0.15 * (signal - float(signal.mean()))
        + rng.normal(0.0, 0.05, size=len(df))
    )
    df["exchcd"] = 1
    return df[["date", "permno", "ret", "me", "exchcd"]]


def test_size_pipeline_integration() -> None:
    crsp = _make_crsp()
    # Dummy Compustat and link tables (unused for size signal)
    funda = pd.DataFrame(
        {
            "gvkey": ["1", "2"],
            "datadate": [pd.Timestamp("2008-12-31"), pd.Timestamp("2009-12-31")],
            "fyear": [2008, 2009],
            "fyr": [12, 12],
            "at": [10.0, 11.0],
        }
    )
    lnkhist = pd.DataFrame(
        {
            "gvkey": ["1", "2"],
            "lpermno": [0, 1],
            "linktype": ["LC", "LU"],
            "linkdt": [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01")],
            "linkenddt": [pd.Timestamp("2099-12-31"), pd.Timestamp("2099-12-31")],
        }
    )
    res = run_pipeline(
        crsp=crsp,
        funda=funda,
        lnkhist=lnkhist,
        n_bins=4,
        nyse_breaks=True,
        min_obs=5,
    )
    summ = res["portfolio_summary"]
    hl = res["hl_series"]
    lambdas = res["fmb_lambdas"]
    se = res["fmb_se"]
    ts = res["fmb_ts"]
    # Summary should have n_bins + 1 rows (including L-S)
    assert len(summ) == 4 + 1
    assert (summ["bin"] == "L‑S").any()
    assert hl is not None and len(hl) > 0
    # Fama–MacBeth output shapes
    assert "signal" in lambdas.index
    assert "const" in lambdas.index
    assert np.isfinite(se["signal"])
    assert ts.shape[0] == len(crsp["date"].unique())

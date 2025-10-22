from __future__ import annotations

import numpy as np
import pandas as pd

from aa.tests.portfolios import SortConfig, univariate_sort


def test_univariate_sort_smoke() -> None:
    rng = np.random.default_rng(123)

    months = pd.date_range("2000-01-31", periods=18, freq="ME")
    permnos = list(range(10001, 10081))
    idx = pd.MultiIndex.from_product(
        [months, pd.Index(permnos)], names=["date", "permno"]
    )
    panel = idx.to_frame(index=False)

    panel["exchcd"] = 1
    panel["me"] = np.exp(rng.normal(10.0, 0.5, size=len(panel)))
    signal = -np.log(panel["me"])
    panel["ret"] = (
        0.01
        + 0.20 * (signal - float(signal.mean()))
        + rng.normal(0.0, 0.05, size=len(panel))
    )

    returns = panel[["date", "permno", "ret"]].copy()
    signal_df = panel[["date", "permno"]].copy()
    signal_df["signal"] = signal
    size = panel[["date", "permno", "me"]].copy()
    exch = panel[["date", "permno", "exchcd"]].copy()

    res = univariate_sort(
        returns=returns,
        signal=signal_df,
        size=size,
        exch=exch,
        config=SortConfig(n_bins=5, nyse_breaks=True, min_obs=50),
    )

    ts = res["time_series"]
    summ = res["summary"]

    # Basic sanity: have bins and L-S row
    assert {"ret_ew", "ret_vw", "bin"}.issubset(ts.columns)
    assert "L-S" in set(summ["bin"].astype(str))

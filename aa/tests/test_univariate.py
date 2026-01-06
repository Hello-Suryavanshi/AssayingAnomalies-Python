"""Tests for the univariate portfolio sorts.

These tests verify that the ``univariate_sort`` function produces
reasonable outputs when given simple synthetic data.  The tests check
for the correct number of summary rows, the presence of a long–short
row, and that the long–short return equals the difference between the
highest and lowest portfolio averages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aa.asset_pricing import univariate_sort, SortConfig


def _make_synthetic_data():
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    records = []
    signal_vals = np.linspace(-1, 1, 10)
    for date in dates:
        for i, s in enumerate(signal_vals):
            ret = 0.05 + 0.1 * s
            me = 100 + 10 * i
            records.append((date, i, s, ret, me, 1))
    df = pd.DataFrame(
        records, columns=["date", "permno", "signal", "ret", "me", "exchcd"]
    )
    returns = df[["date", "permno", "ret"]].copy()
    signal = df[["date", "permno", "signal"]].copy()
    size = df[["date", "permno", "me"]].copy()
    exch = df[["date", "permno", "exchcd"]].copy()
    return returns, signal, size, exch


def test_univariate_sort_basic_functionality():
    returns, signal, size, exch = _make_synthetic_data()
    config = SortConfig(n_bins=5, nyse_breaks=False, min_obs=1)
    res = univariate_sort(
        returns=returns, signal=signal, size=size, exch=exch, config=config
    )
    summ = res["summary"]
    assert len(summ) == config.n_bins + 1
    assert (summ["bin"] == "L‑S").any()
    hl = summ.set_index("bin")
    diff_ew = float(hl.at[5, "ret_ew"] - hl.at[1, "ret_ew"])
    ls_ew = float(hl.at["L‑S", "ret_ew"])
    assert abs(diff_ew - ls_ew) < 1e-12
    diff_vw = float(hl.at[5, "ret_vw"] - hl.at[1, "ret_vw"])
    ls_vw = float(hl.at["L‑S", "ret_vw"])
    assert abs(diff_vw - ls_vw) < 1e-12

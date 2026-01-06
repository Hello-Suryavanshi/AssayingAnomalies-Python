"""Tests for the Fama–MacBeth estimator.

These tests use synthetic data to verify that the ``fama_macbeth``
function returns reasonable coefficients and standard errors.  The test
ensures that when returns are generated as a linear function of a
characteristic plus noise, the estimated risk price is close to the
true coefficient and that the Newey–West standard error is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aa.asset_pricing import fama_macbeth


def _generate_panel(
    T: int = 20, N: int = 30, beta: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = np.arange(200001, 200001 + T)
    data = []
    for t in periods:
        signal = rng.standard_normal(N)
        noise = rng.standard_normal(N) * 0.05
        ret = 0.01 + beta * signal + noise
        permnos = np.arange(N)
        for i in range(N):
            data.append((t, int(permnos[i]), signal[i], ret[i]))
    panel = pd.DataFrame(data, columns=["yyyymm", "permno", "signal", "ret"])
    return panel


def test_fama_macbeth_estimates_close_to_true_value():
    panel = _generate_panel(T=30, N=50, beta=0.7, seed=0)
    lambdas, lambda_ts, se = fama_macbeth(panel, y="ret", xcols=["signal"])
    assert abs(lambdas["signal"] - 0.7) < 0.05
    assert abs(lambdas["const"] - 0.01) < 0.02
    assert float(se["signal"]) > 0
    assert np.isfinite(se["signal"])
    assert len(lambda_ts) == panel["yyyymm"].nunique()
    assert set(lambda_ts.columns) == {"const", "signal"}

import numpy as np
import pandas as pd

from aa.asset_pricing import fama_macbeth


def test_fmb_runs() -> None:
    df = pd.DataFrame(
        {
            "yyyymm": [202001] * 5 + [202002] * 5,
            "ret": np.random.randn(10) / 100,
            "x": np.linspace(-1, 1, 10),
        }
    )
    lambdas, lambda_ts, se = fama_macbeth(df, y="ret", xcols=["x"])

    assert "const" in lambdas.index
    assert "x" in lambdas.index
    assert lambda_ts.shape[0] == 2
    assert se["x"] >= 0

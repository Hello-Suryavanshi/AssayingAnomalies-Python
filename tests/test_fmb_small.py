import pandas as pd
import numpy as np
from aa.tests.fama_macbeth import fama_macbeth


def test_fmb_runs():
    # tiny fake panel
    df = pd.DataFrame(
        {
            "yyyymm": [202001] * 5 + [202002] * 5,
            "ret": np.random.randn(10) / 100,
            "x": np.linspace(-1, 1, 10),
        }
    )
    lambdas, _ = fama_macbeth(df, y="ret", xcols=["x"])
    assert "x" in lambdas.index

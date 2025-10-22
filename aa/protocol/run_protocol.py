from __future__ import annotations
import pandas as pd
from ..tests.portfolios import univariate_sort
from ..tests.fama_macbeth import fama_macbeth


def run(signal_panel: pd.DataFrame, signal_col: str):
    # 1) Portfolio deciles & H-L
    deciles = univariate_sort(signal_panel, signal=signal_col, n=10)
    hl = deciles[deciles["decile"].isin([10, 1])].pivot(
        index="yyyymm", columns="decile", values="ret_vw"
    )
    hl = (hl[10] - hl[1]).rename("hl")

    # 2) Fama–MacBeth cross-section on the characteristic
    lambdas, lambda_ts = fama_macbeth(signal_panel, y="ret", xcols=[signal_col])

    return {"deciles": deciles, "hl": hl, "fmb": lambdas, "fmb_ts": lambda_ts}

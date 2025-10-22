import pandas as pd
import numpy as np


def value_weighted_return(df: pd.DataFrame, ret_col="ret", weight_col="me") -> float:
    w = df[weight_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r = df[ret_col].astype("float64").fillna(0.0)
    denom = w.sum()
    return float((w * r).sum() / denom) if denom else 0.0


def univariate_sort(panel: pd.DataFrame, signal: str, n=10):
    # expects panel with columns: yyyymm, permno, ret, me, signal
    out = []
    for k, grp in panel.groupby("yyyymm"):
        q = pd.qcut(grp[signal].rank(method="first"), n, labels=False) + 1
        g = grp.copy()
        g["decile"] = q
        vw = (
            g.groupby("decile")
            .apply(value_weighted_return)
            .rename("ret_vw")
            .reset_index()
        )
        vw["yyyymm"] = k
        out.append(vw)
    return pd.concat(out, ignore_index=True)

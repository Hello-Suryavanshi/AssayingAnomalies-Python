import pandas as pd
import statsmodels.api as sm


def fama_macbeth(panel: pd.DataFrame, y="ret", xcols=None):
    if xcols is None:
        xcols = []
    # First-pass: time-series betas (here assume x are characteristics -> direct second-pass)
    out = []
    for t, grp in panel.groupby("yyyymm"):
        X = sm.add_constant(grp[xcols])
        yv = grp[y]
        res = sm.OLS(yv, X, missing="drop").fit()
        out.append(res.params.rename(t))
    lambda_ts = pd.DataFrame(out).sort_index()
    lambdas = lambda_ts.mean()
    return lambdas, lambda_ts

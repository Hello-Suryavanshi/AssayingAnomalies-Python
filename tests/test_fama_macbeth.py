import pandas as pd

from aa.asset_pricing.fama_macbeth import fama_macbeth_full


def test_fama_macbeth_full_multifactor():
    # Create a simple panel for 5 firms across 4 periods
    periods = [1, 2, 3, 4]
    records = []
    for t in periods:
        for firm in range(1, 6):
            rec = {
                "yyyymm": t,
                "permno": firm,
                "ret": 0.01 * firm + 0.02 * t,
                "size": float(firm),
                "bm": float(6 - firm),
            }
            records.append(rec)
    panel = pd.DataFrame(records)
    res = fama_macbeth_full(panel, y="ret", xcols=["size", "bm"], time_col="yyyymm")
    # Check keys
    for k in ["lambdas", "lambda_ts", "se", "tstat", "n_obs"]:
        assert k in res
    # There should be an intercept and two coefficients
    assert set(res["lambdas"].index) == {"const", "size", "bm"}
    # n_obs should equal number of periods (4) for each coefficient
    assert (res["n_obs"] == len(periods)).all()

import pandas as pd


def yyyymm_to_period(yyyymm: int) -> pd.Period:
    y, m = divmod(yyyymm, 100)
    return pd.Period(freq="M", year=y, month=m)


def month_end(dt: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(dt).to_period("M") + 1).to_timestamp() - pd.Timedelta(days=1)

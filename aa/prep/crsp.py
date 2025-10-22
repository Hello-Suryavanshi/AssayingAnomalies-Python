import pandas as pd
from ..util.ids import keep_common_equity, normalize_permno


def clean_crsp_msf(msf: pd.DataFrame) -> pd.DataFrame:
    df = msf.copy()
    df = normalize_permno(df)
    df = df.pipe(keep_common_equity)
    # Fix prices: prc negative indicates bid/ask convention; take abs
    if "prc" in df:
        df["prc"] = df["prc"].abs()
    # Market equity (ME) in $ thousands if shrout shares in thousands; align to dollars if needed
    if {"prc", "shrout"}.issubset(df.columns):
        df["me"] = (df["prc"] * df["shrout"]).astype("float64")
    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    return df

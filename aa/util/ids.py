import pandas as pd


def normalize_permno(df: pd.DataFrame) -> pd.DataFrame:
    if "permno" in df:
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


def keep_common_equity(df: pd.DataFrame) -> pd.DataFrame:
    # placeholder: refine using CRSP share codes 10/11 and exchcd in prep/crsp.py
    return df[df["shrcd"].isin([10, 11])]

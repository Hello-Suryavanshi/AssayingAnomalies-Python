import pandas as pd


def key_fields(funda: pd.DataFrame) -> pd.DataFrame:
    df = funda.copy()
    return df[["gvkey", "fyear", "datadate", "at", "ceq", "sale", "cogs", "txdb"]]

from __future__ import annotations
from pathlib import Path
import pandas as pd

CACHE = Path.home() / ".aa_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, relpath: str) -> Path:
    p = CACHE / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read_parquet(relpath: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / relpath)

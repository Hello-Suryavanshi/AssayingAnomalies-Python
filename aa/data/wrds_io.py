# aa/data/wrds_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import wrds  # official WRDS-Py client

from .cache import save_parquet
from ..util.logging import get_logger

log = get_logger("aa.wrds")


@dataclass
class WRDSConfig:
    """Configuration for WRDS pulls."""

    schema_crsp: str = "crsp"
    schema_comp: str = "comp"
    date_start: str = "1962-01-01"
    date_end: Optional[str] = None  # open-ended by default


class WRDSClient:
    """Thin wrapper around wrds.Connection with a few convenience pulls."""

    def __init__(self, cfg: Optional[WRDSConfig] = None) -> None:
        self.cfg = cfg or WRDSConfig()
        # Prompts for credentials if no pgpass; this is the official flow.
        self.conn = wrds.Connection()
        log.info("Connected to WRDS.")

    def pull_crsp_monthly(self) -> pd.DataFrame:
        """Fetch CRSP monthly (msf) and cache to ~/.aa_cache/v0/crsp_msf.parquet."""
        where_end = f"and date <= '{self.cfg.date_end}'" if self.cfg.date_end else ""
        q = f"""
            select permno, permco, date, ret, retx, shrout, prc, vol, shrcd, exchcd
            from {self.cfg.schema_crsp}.msf
            where date >= '{self.cfg.date_start}'
            {where_end}
        """
        log.info("Querying CRSP monthly...")
        df: pd.DataFrame = self.conn.raw_sql(q, date_cols=["date"])
        log.info("CRSP monthly rows: %d", len(df))
        save_parquet(df, "v0/crsp_msf.parquet")
        return df

    def pull_compustat_annual(self) -> pd.DataFrame:
        """Fetch Compustat annual fundamentals (funda) and cache."""
        q = f"""
            select gvkey, fyear, datadate, at, ceq, sale, cogs, txdb
            from {self.cfg.schema_comp}.funda
            where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
              and datadate >= '{self.cfg.date_start}'
        """
        log.info("Querying Compustat annual...")
        df: pd.DataFrame = self.conn.raw_sql(q, date_cols=["datadate"])
        save_parquet(df, "v0/comp_funda.parquet")
        return df

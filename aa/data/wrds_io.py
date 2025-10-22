from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import wrds  # WRDS-Py
from .cache import save_parquet
from ..util.logging import get_logger

log = get_logger("aa.wrds")


@dataclass
class WRDSConfig:
    # You can also rely on interactive login; pgpass is supported by wrds.Connection
    schema_crsp: str = "crsp"
    schema_comp: str = "comp"
    date_start: str = "1962-01-01"
    date_end: str | None = None  # open-ended


class WRDSClient:
    def __init__(self, cfg: WRDSConfig | None = None):
        self.cfg = cfg or WRDSConfig()
        self.conn = wrds.Connection()  # prompts if no pgpass; official flow
        # https://pypi.org/project/wrds/ ; https://wrds-www.wharton.upenn.edu/pages/grid-items/using-python-wrds-platform/
        log.info("Connected to WRDS.")

    def pull_crsp_monthly(self) -> pd.DataFrame:
        q = f"""
        select permno, permco, date, ret, retx, shrout, prc, vol, shrcd, exchcd
        from {self.cfg.schema_crsp}.msf
        where date >= '{self.cfg.date_start}'
        {f"and date <= '{self.cfg.date_end}'" if self.cfg.date_end else ""}
        """
        log.info("Querying CRSP monthly...")
        df = self.conn.raw_sql(q, date_cols=["date"])
        log.info("CRSP monthly rows: %d", len(df))
        save_parquet(df, "v0/crsp_msf.parquet")
        return df

    def pull_compustat_annual(self) -> pd.DataFrame:
        q = f"""
        select gvkey, fyear, datadate, at, ceq, sale, cogs, txdb
        from {self.cfg.schema_comp}.funda
        where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
          and datadate >= '{self.cfg.date_start}'
        """
        log.info("Querying Compustat annual...")
        df = self.conn.raw_sql(q, date_cols=["datadate"])
        save_parquet(df, "v0/comp_funda.parquet")
        return df

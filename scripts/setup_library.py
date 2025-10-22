from __future__ import annotations

from typing import cast
import pandas as pd

from aa.data.wrds_io import WRDSClient, WRDSConfig
from aa.prep.crsp import clean_crsp_msf


def main() -> None:
    wrds = WRDSClient(WRDSConfig(date_start="1962-01-01"))
    msf = cast(pd.DataFrame, wrds.pull_crsp_monthly())
    msf_clean = cast(pd.DataFrame, clean_crsp_msf(msf))
    print(msf_clean.head())


if __name__ == "__main__":
    main()

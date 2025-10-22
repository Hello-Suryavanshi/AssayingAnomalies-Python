# Example “use” script: build a toy signal (size = log(me)), run portfolio & FMB
import numpy as np
from aa.data.cache import read_parquet
from aa.prep.crsp import clean_crsp_msf
from aa.protocol.run_protocol import run
from aa.protocol.reporting import to_markdown_tables


def main():
    msf = read_parquet("v0/crsp_msf.parquet")
    panel = clean_crsp_msf(msf)
    panel["signal"] = np.log(panel["me"].replace(0, np.nan))
    res = run(
        panel.rename(columns={"date": "dt"}).assign(
            yyyymm=lambda d: d["dt"].dt.year * 100 + d["dt"].dt.month
        ),
        signal_col="signal",
    )
    print(to_markdown_tables(res))


if __name__ == "__main__":
    main()

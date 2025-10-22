# Plug in any custom signal CSV (permno, yyyymm, signal), merge to CRSP panel, run protocol
import pandas as pd
from aa.data.cache import read_parquet
from aa.prep.crsp import clean_crsp_msf
from aa.protocol.run_protocol import run
from aa.protocol.reporting import to_markdown_tables


def main():
    msf = clean_crsp_msf(read_parquet("v0/crsp_msf.parquet"))
    signal = pd.read_csv("examples/my_signal.csv")  # user-provided
    panel = msf.merge(signal, on=["permno", "yyyymm"], how="inner").rename(
        columns={"date": "dt"}
    )
    res = run(
        panel.assign(yyyymm=lambda d: d["dt"].dt.year * 100 + d["dt"].dt.month),
        "signal",
    )
    print(to_markdown_tables(res))


if __name__ == "__main__":
    main()

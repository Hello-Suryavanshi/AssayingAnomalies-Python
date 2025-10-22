from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)

    # --- Tiny CRSP monthly panel ---
    months = pd.date_range("2000-01-31", periods=24, freq="M")
    permnos = list(range(10001, 10051))  # plain list keeps mypy happy
    idx = pd.MultiIndex.from_product(
        [months, pd.Index(permnos)], names=["date", "permno"]
    )
    crsp = idx.to_frame(index=False)
    crsp["exchcd"] = 1  # NYSE for simplicity
    crsp["me"] = np.exp(rng.normal(10.0, 0.6, size=len(crsp)))  # positive sizes
    signal = -np.log(crsp["me"])
    crsp["ret"] = (
        0.01 + 0.25 * (signal - signal.mean()) + rng.normal(0.0, 0.05, size=len(crsp))
    )
    (out_dir / "crsp_msf.parquet").unlink(missing_ok=True)
    crsp.to_parquet(out_dir / "crsp_msf.parquet", index=False)

    # --- Tiny FUNDA ---
    gvkeys = np.array([str(1000 + i) for i in range(50)])
    datadates = pd.to_datetime(["1999-12-31", "2000-12-31", "2001-12-31"])
    rows = [
        (g, dd, dd.year, 12, float(np.exp(rng.normal(12.0, 0.4))))
        for g in gvkeys
        for dd in datadates
    ]
    funda = pd.DataFrame(rows, columns=["gvkey", "datadate", "fyear", "fyr", "at"])
    (out_dir / "comp_funda.parquet").unlink(missing_ok=True)
    funda.to_parquet(out_dir / "comp_funda.parquet", index=False)

    # --- Tiny CCM linkhist (1:1 mapping gvkey→permno) ---
    lnk = pd.DataFrame(
        {
            "gvkey": gvkeys,
            "lpermno": permnos,
            "linktype": ["LC"] * len(gvkeys),
            "linkdt": pd.Timestamp("1990-01-01"),
            "linkenddt": pd.NaT,
        }
    )
    (out_dir / "ccm_lnkhist.parquet").unlink(missing_ok=True)
    lnk.to_parquet(out_dir / "ccm_lnkhist.parquet", index=False)

    print(f"Wrote tiny cache to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
Demo: load data (WRDS or cache) → prepare Compustat → CCM link → -log(ME) sorts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional
import pandas as pd

from aa.prep.compustat import prepare_compustat_annual
from aa.prep.linktables import map_gvkey_to_permno

# Optional engines: declare as optionals (no type: ignore needed)
univariate_sort: Optional[Callable[..., Dict[str, pd.DataFrame]]] = None
SortConfig: Any = None

try:
    from aa.tests.portfolios import (
        univariate_sort as _univariate_sort,
        SortConfig as _SortConfig,
    )

    univariate_sort = _univariate_sort
    SortConfig = _SortConfig
except Exception:
    pass


def _try_wrds_loader() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from aa.data import wrds_io as wio  # runtime optional

        if hasattr(wio, "get_crsp_monthly"):
            out["crsp"] = wio.get_crsp_monthly()
        if hasattr(wio, "get_compustat_funda"):
            out["funda"] = wio.get_compustat_funda()
        if hasattr(wio, "get_ccm_linkhist"):
            out["lnkhist"] = wio.get_ccm_linkhist()
    except Exception:
        pass
    return out


def _try_home_cache() -> Dict[str, Any]:
    try:
        from aa.data.cache import read_parquet  # runtime optional
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    try:
        out["crsp"] = read_parquet("v0/crsp_msf.parquet")
        out["funda"] = read_parquet("v0/comp_funda.parquet")
        out["lnkhist"] = read_parquet("v0/ccm_lnkhist.parquet")
        return out
    except Exception:
        return {}


def _try_local_parquets() -> Dict[str, Any]:
    roots = [Path("data/cache"), Path("aa/data/cache"), Path("cache")]
    names = {
        "crsp": "crsp_msf.parquet",
        "funda": "comp_funda.parquet",
        "lnkhist": "ccm_lnkhist.parquet",
    }
    out: Dict[str, Any] = {}
    for r in roots:
        if not r.exists():
            continue
        for k, fname in names.items():
            p = r / fname
            if p.exists():
                out[k] = pd.read_parquet(p)
    return out


def _normalize_crsp(df: pd.DataFrame) -> pd.DataFrame:
    m: dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"date", "permno", "ret", "me", "exchcd"}:
            m[c] = lc
    g = df.rename(columns=m).copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce").dt.tz_localize(None)
    cols = ["date", "permno", "ret", "me"]
    if "exchcd" in g.columns:
        cols.append("exchcd")
    return g[cols]


def main() -> None:
    loaders = _try_wrds_loader() or _try_home_cache() or _try_local_parquets()
    if not {"crsp", "funda", "lnkhist"}.issubset(loaders):
        raise RuntimeError(
            "Missing inputs. Provide CRSP, FUNDA, and CCM LNKHIST via aa.data.wrds_io, "
            "~/.aa_cache/v0/*.parquet, or data/cache/*.parquet."
        )

    crsp = _normalize_crsp(loaders["crsp"])
    funda_raw = loaders["funda"]
    lnkhist = loaders["lnkhist"]

    # Prepare Compustat and (optionally) link to PERMNO at assign_month (kept for later steps)
    funda = prepare_compustat_annual(funda_raw)
    _linked = map_gvkey_to_permno(funda, lnkhist, date_col="assign_month")

    # Build simple signal: -log(ME) lagged 1 month
    crsp = crsp.sort_values(["permno", "date"]).copy()
    crsp["me_lag"] = crsp.groupby("permno")["me"].shift(1)
    signal = (
        crsp[["date", "permno", "me_lag"]].rename(columns={"me_lag": "signal"}).copy()
    )
    signal["signal"] = -signal["signal"].astype(float)

    returns = crsp[["date", "permno", "ret"]].copy()
    size = crsp[["date", "permno", "me"]].copy()
    exch = (
        crsp[["date", "permno", "exchcd"]].copy() if "exchcd" in crsp.columns else None
    )

    if univariate_sort is None or SortConfig is None:
        print("Portfolio engine not available. Skipping sorts.")
        return

    res = univariate_sort(
        returns=returns,
        signal=signal,
        size=size,
        exch=exch,
        config=SortConfig(n_bins=5, nyse_breaks=True, min_obs=50),
    )
    summ = res["summary"]

    print("# Portfolio Sorts (EW/VW) on -log(ME)")
    print("## Summary (avg monthly returns by bin; L-S is K−1):")
    print(summ.to_markdown(index=False))


if __name__ == "__main__":
    main()

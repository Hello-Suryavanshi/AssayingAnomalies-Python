from __future__ import annotations

from typing import List
import pandas as pd

__all__ = ["map_gvkey_to_permno"]


def _normalize_linkhist(lnk: pd.DataFrame) -> pd.DataFrame:
    """Clean CCM link history; normalize date bounds and allowed link types."""
    keep_types: List[str] = ["LC", "LU", "LX", "LS"]
    out = lnk.copy()

    # Basic cols
    for c in ["gvkey", "lpermno", "linktype", "linkdt", "linkenddt"]:
        if c not in out.columns:
            raise KeyError(f"lnk is missing column: {c}")

    out = out.loc[out["linktype"].isin(keep_types)].copy()
    out["linkdt"] = pd.to_datetime(out["linkdt"], errors="coerce")
    # Treat open-ended as 2099-12-31
    out["linkenddt"] = pd.to_datetime(out["linkenddt"], errors="coerce")
    out["linkenddt"] = out["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    # Ensure permno is int
    out["lpermno"] = pd.to_numeric(out["lpermno"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["gvkey", "lpermno", "linkdt"])

    # Resolve overlaps by keeping longest span, then latest start.
    # Use numpy datetime64 arrays so mypy doesn't see ExtensionArray unions.
    start = out["linkdt"].to_numpy("datetime64[ns]")
    end = out["linkenddt"].to_numpy("datetime64[ns]")
    span = (end - start).astype("timedelta64[D]").astype("int64")
    out = out.assign(_span_days=span)

    out = (
        out.sort_values(
            ["gvkey", "lpermno", "_span_days", "linkdt"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["gvkey", "lpermno"], keep="first")
        .reset_index(drop=True)
    )
    return out


def map_gvkey_to_permno(
    funda: pd.DataFrame,
    lnkhist: pd.DataFrame,
    *,
    date_col: str = "assign_month",
) -> pd.DataFrame:
    """
    Map Compustat gvkey rows to CRSP permno using CCM link history with robust date logic.

    Parameters
    ----------
    funda : DataFrame
        Must contain columns: ['gvkey', date_col]
    lnkhist : DataFrame
        CCM link history with columns ['gvkey','lpermno','linktype','linkdt','linkenddt']
    date_col : str
        The column in `funda` used as the 'as-of' date for the link (e.g., June assignment month).

    Returns
    -------
    DataFrame
        `funda` with an added 'permno' (Int64) column for rows where a valid link exists.
    """
    if "gvkey" not in funda.columns:
        raise KeyError("funda must have column 'gvkey'")
    if date_col not in funda.columns:
        raise KeyError(f"funda must have date column '{date_col}'")

    lnk = _normalize_linkhist(lnkhist)

    df = funda.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Merge and filter by active window: linkdt <= asof <= linkenddt
    merged = df.merge(
        lnk[["gvkey", "lpermno", "linkdt", "linkenddt", "_span_days"]],
        on="gvkey",
        how="left",
        validate="m:m",
    )
    asof = merged[date_col].to_numpy("datetime64[ns]")
    start = merged["linkdt"].to_numpy("datetime64[ns]")
    end = merged["linkenddt"].to_numpy("datetime64[ns]")

    active_mask = (asof >= start) & (asof <= end)
    active = merged.loc[active_mask].copy()

    # If multiple active rows remain for a gvkey/asof, keep longest span, then latest start
    active = (
        active.sort_values(
            ["gvkey", date_col, "_span_days", "linkdt"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["gvkey", date_col], keep="first")
        .reset_index(drop=True)
    )

    # Attach permno back
    out = df.merge(
        active[["gvkey", date_col, "lpermno"]],
        on=["gvkey", date_col],
        how="left",
        validate="1:1",
    )
    out = out.rename(columns={"lpermno": "permno"})
    out["permno"] = out["permno"].astype("Int64")
    return out

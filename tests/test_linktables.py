from __future__ import annotations

import pandas as pd
from aa.prep.linktables import clean_ccm_linkhist, map_gvkey_to_permno


def test_ccm_interval_join_minimal() -> None:
    # Fake FUNDA (already assigned to June Y+1)
    comp = pd.DataFrame(
        {
            "gvkey": ["1001", "1001", "2002"],
            "assign_month": pd.to_datetime(["2001-06-30", "2002-06-30", "2001-06-30"]),
            "at": [10.0, 12.0, 5.0],
        }
    )

    # Fake LNKHIST: gvkey 1001 links to permno 50001 from 2000-01-01..2001-12-31, then 60002
    lh = pd.DataFrame(
        {
            "gvkey": ["1001", "1001", "2002"],
            "lpermno": [50001, 60002, 70003],
            "linktype": ["LC", "LC", "LU"],
            "linkdt": pd.to_datetime(["2000-01-01", "2002-01-01", "2000-01-01"]),
            "linkenddt": [
                pd.Timestamp("2001-12-31"),
                pd.NaT,
                pd.Timestamp("2001-12-31"),
            ],
        }
    )

    out = map_gvkey_to_permno(comp, lh, date_col="assign_month")
    # 1001-2001 should map to 50001; 1001-2002 to 60002; 2002-2001 maps (ends 2001) to 70003
    assert (
        out.loc[
            (out.gvkey == "1001") & (out.assign_month == pd.Timestamp("2001-06-30")),
            "permno",
        ].item()
        == 50001
    )
    assert (
        out.loc[
            (out.gvkey == "1001") & (out.assign_month == pd.Timestamp("2002-06-30")),
            "permno",
        ].item()
        == 60002
    )
    assert (
        out.loc[
            (out.gvkey == "2002") & (out.assign_month == pd.Timestamp("2001-06-30")),
            "permno",
        ].item()
        == 70003
    )

    # Ensure cleaning keeps only LC/LU and fills open-ended end dates
    cleaned = clean_ccm_linkhist(lh)
    assert set(cleaned["linktype"].unique()).issubset({"LC", "LU"})
    assert cleaned["linkenddt"].notna().all()

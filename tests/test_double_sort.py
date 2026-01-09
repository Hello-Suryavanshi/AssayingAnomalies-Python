import pandas as pd

from aa.asset_pricing.double_sort import double_sort, DoubleSortConfig


def test_double_sort_returns_shape():
    # Synthetic data for one month with six firms
    date = pd.to_datetime("2020-01-31")
    returns = pd.DataFrame(
        {
            "date": [date] * 6,
            "permno": list(range(1, 7)),
            "ret": [0.01, -0.02, 0.03, -0.01, 0.00, 0.02],
        }
    )
    signal1 = pd.DataFrame(
        {
            "date": [date] * 6,
            "permno": list(range(1, 7)),
            "signal": [1, 2, 3, 4, 5, 6],
        }
    )
    signal2 = pd.DataFrame(
        {
            "date": [date] * 6,
            "permno": list(range(1, 7)),
            "signal": [6, 5, 4, 3, 2, 1],
        }
    )
    size = pd.DataFrame(
        {
            "date": [date] * 6,
            "permno": list(range(1, 7)),
            "me": [10, 20, 30, 40, 50, 60],
        }
    )
    res = double_sort(
        returns=returns,
        signal_1=signal1,
        signal_2=signal2,
        size=size,
        exch=None,
        config=DoubleSortConfig(
            n_bins_1=2, n_bins_2=3, nyse_breaks=False, min_obs=2, conditional=False
        ),
    )
    # Summary should not be empty and should not exceed the total number of bins
    max_bins = 2 * 3
    summary_rows = res["summary"].shape[0]
    assert summary_rows >= 1 and summary_rows <= max_bins
    # Time‑series should have as many rows as there are observed portfolios
    # (one per bin in this synthetic month)
    assert res["time_series"].shape[0] == summary_rows
    # High–low tables contain at most one row (since there is one date)
    assert res["hl_dim1"].shape[0] <= 1
    assert res["hl_dim2"].shape[0] <= 1

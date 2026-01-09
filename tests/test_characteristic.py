import pandas as pd

from aa.asset_pricing.characteristic import characteristic_managed_portfolio


def test_characteristic_managed_portfolio_basic():
    # Two periods, three firms, one characteristic
    returns = pd.DataFrame(
        {
            "date": [
                "2020-01-31",
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-02-29",
            ],
            "permno": [1, 2, 3, 1, 2, 3],
            "ret": [0.01, -0.02, 0.03, 0.05, 0.00, -0.01],
        }
    )
    char = pd.DataFrame(
        {
            "date": [
                "2020-01-31",
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-02-29",
            ],
            "permno": [1, 2, 3, 1, 2, 3],
            "size": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        }
    )
    out = characteristic_managed_portfolio(returns, char)
    # Should produce one row per date
    assert out.shape[0] == returns["date"].nunique()
    # Column name should be 'size'
    assert "size" in out.columns
    # Returns should not all be NaN
    assert out["size"].notna().all()

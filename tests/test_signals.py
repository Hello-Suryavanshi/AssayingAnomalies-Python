import pandas as pd

from aa.signals.size import compute_size_signal
from aa.signals.book_to_market import compute_book_to_market_signal
from aa.signals.momentum import compute_momentum_signal
from aa.signals.investment import compute_investment_signal
from aa.signals.profitability import compute_profitability_signal


def test_size_signal_basic():
    # Create synthetic market cap data for two firms across two dates
    df = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "permno": [1, 2, 1, 2],
            "me": [100.0, 400.0, 110.0, 380.0],
        }
    )
    out = compute_size_signal(df)
    # Ensure output has same number of rows
    assert len(out) == len(df)
    # First month signals should be NaN due to lagging
    first_date = pd.to_datetime("2020-01-31")
    assert out.loc[out["date"] == first_date, "signal"].isna().all()
    # Second month signals should reflect negative log of prior month ME
    second_date = pd.to_datetime("2020-02-29")
    out_feb = out[out["date"] == second_date]
    sig1 = out_feb.loc[out_feb["permno"] == 1, "signal"].iloc[0]
    sig2 = out_feb.loc[out_feb["permno"] == 2, "signal"].iloc[0]
    # Firm 1 has smaller ME in prior month (100 vs 400), so -log(100) > -log(400)
    assert sig1 > sig2


def test_book_to_market_returns_dataframe():
    # Minimal CRSP & Compustat data to run book‑to‑market signal
    # Provide market equity directly as required by compute_book_to_market_signal
    crsp = pd.DataFrame(
        {
            "date": ["2020-06-30", "2020-06-30"],
            "permno": [1, 2],
            "me": [100.0, 200.0],
        }
    )
    comp = pd.DataFrame(
        {
            "datadate": ["2019-12-31", "2019-12-31"],
            "permno": [1, 2],
            "be": [50.0, 100.0],
        }
    )
    # In this test we intentionally ignore real lag logic and pass the minimal dataset
    out = compute_book_to_market_signal(crsp=crsp, funda=comp)
    # Output should have two rows and columns date, permno, signal
    assert out.shape[0] == 2
    assert set(["date", "permno", "signal"]).issubset(out.columns)


def test_momentum_signal_shape():
    # Create returns for 2 firms over 15 months
    dates = pd.date_range("2020-01-31", periods=15, freq="M")
    panel = pd.DataFrame(
        [(d, p, 0.01 * (i + p)) for i, d in enumerate(dates) for p in [1, 2]],
        columns=["date", "permno", "ret"],
    )
    out = compute_momentum_signal(panel)
    # There should be len(dates) rows for each firm
    assert out["date"].nunique() == len(dates)
    # Signal column present
    assert "signal" in out.columns


def test_investment_signal_non_empty():
    # Synthetic book assets data
    comp = pd.DataFrame(
        {
            "datadate": ["2018-12-31", "2018-12-31", "2019-12-31", "2019-12-31"],
            "permno": [1, 2, 1, 2],
            "at": [100.0, 200.0, 110.0, 190.0],
        }
    )
    # Provide both CRSP and Compustat dataframes.  Only dates and permno
    # are needed in CRSP for alignment.
    crsp = pd.DataFrame(
        {
            "date": ["2019-12-31", "2019-12-31"],
            "permno": [1, 2],
        }
    )
    out = compute_investment_signal(crsp=crsp, funda=comp)
    # Should return at most one observation per firm per date
    assert not out.empty
    assert set(["date", "permno", "signal"]).issubset(out.columns)


def test_profitability_signal_non_empty():
    # Provide 'op' (operating profitability numerator) directly
    comp = pd.DataFrame(
        {
            "datadate": ["2019-12-31", "2019-12-31"],
            "permno": [1, 2],
            "op": [40.0, 70.0],
            "be": [80.0, 150.0],
        }
    )
    crsp = pd.DataFrame(
        {
            "date": ["2020-06-30", "2020-06-30"],
            "permno": [1, 2],
        }
    )
    out = compute_profitability_signal(crsp=crsp, funda=comp)
    assert not out.empty
    assert "signal" in out.columns

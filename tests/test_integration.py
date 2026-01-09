import pandas as pd
import numpy as np

from aa.signals.size import compute_size_signal
from aa.signals.book_to_market import compute_book_to_market_signal
from aa.signals.momentum import compute_momentum_signal

from aa.asset_pricing.double_sort import double_sort, DoubleSortConfig
from aa.asset_pricing.fama_macbeth import fama_macbeth_full


def test_integration_end_to_end():
    """
    End‑to‑end test that combines signals, double sorts and
    Fama–MacBeth regressions.  Uses synthetic data so the primary goal
    is that the functions interoperate and return objects of the
    expected shapes.
    """
    # Generate synthetic panel for 12 months and 10 firms
    dates = pd.date_range("2020-01-31", periods=12, freq="M")
    firms = list(range(1, 11))
    rows = []
    me_rows = []
    exch_rows = []
    for d in dates:
        for f in firms:
            # Assign deterministic returns that vary by firm and date
            ret = 0.01 + 0.005 * ((f % 3) - 1)
            rows.append({"date": d, "permno": f, "ret": ret})
            me_rows.append({"date": d, "permno": f, "me": float(f * 100)})
            exch_rows.append({"date": d, "permno": f, "exchcd": 1})
    returns = pd.DataFrame(rows)
    size_df = pd.DataFrame(me_rows)
    exch_df = pd.DataFrame(exch_rows)
    # Size signal (neg log market equity)
    size_signal_df = compute_size_signal(size_df)
    # Book‑to‑market: create pseudo price and book equity
    # Provide market equity directly as required by compute_book_to_market_signal
    # Book-to-market: create pseudo price and book equity
    # Provide market equity directly as required by compute_book_to_market_signal
    crsp = pd.DataFrame(
        {
            "date": dates.repeat(len(firms)),
            "permno": np.tile(firms, len(dates)),
            # vary by firm so bm varies cross-sectionally
            "me": np.tile([float(f * 50) for f in firms], len(dates)),
        }
    )

    comp = pd.DataFrame(
        {
            "datadate": pd.to_datetime(
                [d - pd.DateOffset(months=6) for d in dates]
            ).repeat(len(firms)),
            "permno": np.tile(firms, len(dates)),
            # can stay constant; bm varies because me varies
            "be": 100.0,
        }
    )

    bm_signal_df = compute_book_to_market_signal(crsp=crsp, funda=comp)

    bm_signal_df = compute_book_to_market_signal(crsp=crsp, funda=comp)
    # Momentum signal
    mom_signal_df = compute_momentum_signal(returns)
    assert "signal" in mom_signal_df.columns
    # Perform a double sort on size and book‑to‑market
    res = double_sort(
        returns=returns,
        signal_1=size_signal_df.rename(columns={"signal": "signal"}),
        signal_2=bm_signal_df.rename(columns={"signal": "signal"}),
        size=size_df,
        exch=exch_df,
        config=DoubleSortConfig(
            n_bins_1=3, n_bins_2=3, nyse_breaks=False, min_obs=20, conditional=False
        ),
    )
    # Summary should not be empty
    assert not res["summary"].empty
    # Fama–MacBeth regression on returns using size and bm signals
    panel = returns.merge(
        size_signal_df[["date", "permno", "signal"]], on=["date", "permno"], how="left"
    )
    panel = panel.rename(columns={"signal": "size"})
    panel = panel.merge(
        bm_signal_df[["date", "permno", "signal"]], on=["date", "permno"], how="left"
    )
    panel = panel.rename(columns={"signal": "bm"})
    fm = fama_macbeth_full(panel, y="ret", xcols=["size", "bm"], time_col="date")
    # Verify result has expected keys
    assert set(fm.keys()) == {"lambdas", "lambda_ts", "se", "tstat", "n_obs"}

import unittest

import numpy as np
import pandas as pd

from aa.asset_pricing.univariate import univariate_sort, SortConfig
from aa.regime import run_by_regime
from aa.robustness import RobustnessConfig, run_robustness_checks
from aa.diagnostics import (
    rolling_mean,
    rolling_regression,
    chow_test,
    subsample_fama_macbeth,
    cusum_test,
)
from aa.multiple_testing import (
    bonferroni_correction,
    benjamini_hochberg,
    adjust_pvalues,
    fdr_table,
)
from aa.simulation import (
    random_signal_like,
    permutation_test,
    bootstrap_placebo,
    simulate_null_distribution,
)
from aa.reporting.stability_tables import (
    stability_table,
    robustness_table,
    null_distribution_summary,
)


class TestMilestone4(unittest.TestCase):
    def setUp(self):
        # Create small synthetic data for tests
        dates = pd.date_range("2020-01-31", periods=6, freq="M")
        self.returns = pd.DataFrame(
            {
                "date": dates.repeat(3),
                "permno": [1, 2, 3] * len(dates),
                "ret": [
                    0.01,
                    0.02,
                    -0.01,
                    0.03,
                    0.04,
                    0.02,
                    0.01,
                    -0.02,
                    0.03,
                    0.02,
                    0.05,
                    -0.03,
                    0.04,
                    0.01,
                    0.02,
                    -0.01,
                    0.03,
                    0.06,
                ],
            }
        )
        self.signal = self.returns[["date", "permno"]].copy()
        self.signal["signal"] = [0.2, -0.1, 0.3] * len(dates)
        self.size = self.returns[["date", "permno"]].copy()
        self.size["me"] = [50, 100, 150] * len(dates)
        # Regime indicator: first half vs second half
        self.regimes = pd.DataFrame(
            {
                "date": dates,
                "regime": ["early"] * 3 + ["late"] * 3,
            }
        )

    def test_run_by_regime(self):
        res = run_by_regime(
            regime_indicator=self.regimes,
            analysis_fn=univariate_sort,
            returns=self.returns,
            signal=self.signal,
            size=self.size,
            config=SortConfig(n_bins=2),
        )
        self.assertIn("early", res)
        self.assertIn("late", res)
        self.assertIsInstance(res["early"], dict)

    def test_run_robustness_checks(self):
        # Two configs: EW and VW
        cfgs = [
            RobustnessConfig(weighting="EW", nyse_breaks=False, winsorize=None, lag=0),
            RobustnessConfig(weighting="VW", nyse_breaks=False, winsorize=None, lag=0),
        ]

        def metric_fn(res):
            summ = res["summary"]
            row = summ.loc[summ["bin"] == "L‑S", "ret_ew"]
            return float(row.iloc[0]) if not row.empty else float("nan")

        table = run_robustness_checks(
            analysis_fn=univariate_sort,
            base_config=SortConfig(n_bins=2),
            config_list=cfgs,
            metric_fn=metric_fn,
            returns=self.returns,
            signal=self.signal,
            size=self.size,
            exch=None,
        )
        self.assertEqual(len(table), 2)
        self.assertIn("metric", table.columns)

    def test_diagnostics(self):
        # Rolling mean
        series = pd.Series([1, 2, 3, 4, 5])
        rm = rolling_mean(series, window=3)
        self.assertTrue(np.isnan(rm.iloc[1]))
        # Rolling regression
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        rr = rolling_regression(y, X, window=3)
        self.assertEqual(rr.shape[0], 2)
        # Chow test on simple line with break
        y_data = np.concatenate([np.arange(10), np.arange(5, 15)])
        X_data = np.arange(20).reshape(-1, 1)
        f_stat, p_val = chow_test(y_data, X_data, break_index=10)
        self.assertTrue(f_stat > 0)
        self.assertTrue(0 <= p_val <= 1)
        # Subsample FM
        panel = pd.DataFrame(
            {
                "yyyymm": [1, 1, 2, 2, 3, 3, 4, 4],
                "ret": [0.1, 0.2, 0.2, 0.3, 0.0, -0.1, 0.05, 0.15],
                "x": [1, 2, 1, 2, 1, 2, 1, 2],
            }
        )
        res = subsample_fama_macbeth(
            panel, break_time=3, y="ret", xcols=["x"], time_col="yyyymm"
        )
        self.assertIn("diff", res)
        # CUSUM
        coef_series = pd.Series([0.1, 0.2, 0.15, 0.3, 0.25])
        stat = cusum_test(coef_series)
        self.assertTrue(stat > 0)

    def test_multiple_testing(self):
        p = [0.01, 0.04, 0.1, 0.2]
        adj, cutoff = bonferroni_correction(p, alpha=0.05)
        self.assertEqual(cutoff, 0.05 / len(p))
        rejects_bh = benjamini_hochberg(p, q=0.05)
        self.assertEqual(len(rejects_bh), len(p))
        adj_bh = adjust_pvalues(p, method="BH")
        self.assertEqual(len(adj_bh), len(p))
        table = fdr_table(p)
        self.assertTrue("p_adj_bonf" in table.columns)

    def test_simulation(self):
        rand_sig = random_signal_like(self.signal, method="gaussian", seed=1)
        self.assertEqual(len(rand_sig), len(self.signal))

        # Define a simple analysis and metric
        def simple_analysis(**kwargs):
            # Return mean signal
            sig = kwargs["signal"]["signal"].mean()
            return {"mean": sig}

        def metric_fn(res):
            return res["mean"]

        perm_dist = permutation_test(
            analysis_fn=simple_analysis,
            n_iter=5,
            metric_fn=metric_fn,
            signal_df_key="signal",
            signal=self.signal,
        )
        self.assertEqual(len(perm_dist), 5)
        boot_dist = bootstrap_placebo(
            analysis_fn=simple_analysis,
            n_iter=5,
            metric_fn=metric_fn,
            signal_df_key="signal",
            signal=self.signal,
        )
        self.assertEqual(len(boot_dist), 5)
        null_dist = simulate_null_distribution(
            analysis_fn=simple_analysis,
            n_iter=5,
            metric_fn=metric_fn,
            signal_df_key="signal",
            method="gaussian",
            signal=self.signal,
        )
        self.assertEqual(len(null_dist), 5)

    def test_reporting_stability(self):
        # create dummy results_by_regime
        results = {
            "early": {"summary": pd.DataFrame({"bin": ["L‑S"], "ret_ew": [0.02]})},
            "late": {"summary": pd.DataFrame({"bin": ["L‑S"], "ret_ew": [0.03]})},
        }

        def metric_fn(res):
            summ = res["summary"]
            return float(summ.loc[summ["bin"] == "L‑S", "ret_ew"].iloc[0])

        table = stability_table(results, metric_fn)
        self.assertIn("markdown", table)
        # Robustness table
        robust_df = pd.DataFrame(
            {
                "weighting": ["EW", "VW"],
                "nyse_breaks": [False, False],
                "winsorize": [None, None],
                "lag": [0, 0],
                "holding_period": [1, 1],
                "metric": [0.02, 0.03],
            }
        )
        tab2 = robustness_table(robust_df)
        self.assertIn("latex", tab2)
        # Null distribution summary
        summary = null_distribution_summary([0.1, 0.2, -0.05, 0.15], observed=0.2)
        self.assertIn("markdown", summary)


if __name__ == "__main__":
    unittest.main()

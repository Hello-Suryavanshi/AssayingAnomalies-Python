# Porting Notes for Milestone 4

This document outlines the additional functionality introduced in
Milestone 4 of the Python port of the **Assaying Anomalies** library
relative to the original MATLAB code.  Earlier milestones covered
univariate and double sorts, Fama–MacBeth regressions and basic
reporting utilities.  Milestone 4 elevates the toolkit to a
research‑grade platform by incorporating robustness checks, regime
analysis, stability diagnostics and multiple‑testing controls.

## Coverage relative to MATLAB

The original MATLAB library (
[Novy‑Marx & Velikov](https://github.com/velikov-mihail/AssayingAnomalies))
focuses on constructing and evaluating anomaly portfolios.  It does
not provide off‑the‑shelf routines for sub‑sample analysis, statistical
stability tests or false discovery control.  Milestone 4 fills this
gap by adding the following modules, all of which are absent in the
MATLAB code:

| Module                     | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| `aa.regime`               | Run any analysis (sorts or regressions) by time subsample or market regime. |
| `aa.robustness`           | Standardised robustness toolkit covering weighting schemes, NYSE vs all‑stock breakpoints, winsorisation and signal lags. |
| `aa.diagnostics`          | Rolling means/regressions, Chow structural break tests【581989950563653†L165-L186】, subsample Fama–MacBeth and CUSUM instability measures. |
| `aa.multiple_testing`     | Bonferroni and Benjamini–Hochberg procedures for controlling family‑wise error rate and false discovery rate【239467018970311†L594-L606】. |
| `aa.simulation`           | Random signal generation, permutation tests and bootstrap placebo experiments. |
| `aa.reporting.stability_tables` | Reporting helpers for comparing results across regimes and robustness configurations and for summarising null distributions. |

These additions make the Python implementation a superset of the MATLAB
functionality.  No features from the original code have been removed.

## New diagnostics not in MATLAB

* **Chow structural break test** – implemented in `aa.diagnostics.chow_test`.  It computes an F‑statistic to test equality of regression coefficients across two subsamples【581989950563653†L165-L186】.
* **Rolling alphas and regressions** – available via `aa.diagnostics.rolling_mean` and `rolling_regression`.  These functions allow users to visualise the evolution of portfolio returns or factor loadings.
* **CUSUM parameter instability measure** – `aa.diagnostics.cusum_test` computes the maximum cumulative sum of demeaned coefficients, a simple yet effective indicator of structural change.
* **Multiple testing corrections** – `aa.multiple_testing` provides Bonferroni and Benjamini–Hochberg adjustments and summary tables.  The BH procedure is defined as rejecting hypotheses with p‑values satisfying `p_(r) ≤ (r/m) q`【239467018970311†L594-L606】.
* **Simulation and placebo framework** – `aa.simulation` enables permutation and bootstrap tests as well as synthetic signals drawn from Gaussian noise to gauge the distribution of test statistics under the null.

## Recommended workflow

1. **Design the analysis** using the existing sorts or regressions.  Document all assumptions explicitly in code and write docstrings accordingly.
2. **Perform baseline analysis** on the full sample.
3. **Run regime analysis** using `aa.regime.run_by_regime` to check whether results differ across time subsamples (e.g. pre/post 2000, high/low volatility periods, NBER expansions vs recessions).
4. **Execute robustness checks** with `aa.robustness.run_robustness_checks` over a grid of specifications (EW vs VW, NYSE vs all stocks, winsorisation levels, signal lags).  Summarise the outcomes using `aa.reporting.stability_tables.robustness_table`.
5. **Assess statistical stability** by computing rolling alphas or running `chow_test` and `cusum_test` on the coefficient time series from Fama–MacBeth regressions.
6. **Control for multiple testing** when evaluating a panel of anomalies by adjusting p‑values with `aa.multiple_testing` and creating FDR tables using `aa.multiple_testing.fdr_table`.
7. **Benchmark against null models** by running permutation or bootstrap placebo experiments via `aa.simulation` and comparing the realised statistic to the empirical null distribution using `aa.reporting.stability_tables.null_distribution_summary`.

## Limitations and future work

* The current implementation assumes monthly data and uses simple
  quantile‑based binning.  Extending to daily or intra‑day data will
  require more careful alignment of regimes and signals.
* Holding period adjustments are included in the robustness config but
  not yet implemented.  Future versions may support overlapping
  portfolio holding periods as in standard long–short strategies.
* Some diagnostic tests (e.g. CUSUM) are rudimentary and do not
  compute formal p‑values.  Researchers should complement them with
  more comprehensive stability analyses where appropriate.

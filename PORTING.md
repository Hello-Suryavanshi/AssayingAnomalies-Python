# Porting Notes for Milestone 3

This document tracks the progress of porting the MATLAB **Assaying
Anomalies** toolkit into a research‑grade Python library.  It
documents which features have been fully matched, approximated, or
deferred relative to the original MATLAB code.  Users can consult
this file to understand the methodological coverage of the Python
version and any known numerical differences arising from the port.

## Signal Coverage

| Signal | MATLAB equivalent | Status | Notes |
|-------|------------------|--------|------|
| **Size** | `makeSizeSignal` | ✅ Complete | Negative logarithm of lagged market equity. Matches MATLAB timing and lag structure. |
| **Book‑to‑Market** | `makeBMsignal` | ✅ Complete | Computes `log(be/me)` using book equity at fiscal year *t−1* with a six‑month reporting lag【365329894510713†L40-L52】. Market equity taken contemporaneously from CRSP. |
| **Momentum (12‑2)** | `makeMomSignal` | ✅ Complete | Cumulative return from month *t−12* to *t−2*【696072703907323†L31-L47】. Uses log returns to avoid underflow and aligns CRSP dates. |
| **Investment** | `makeInvSignal` | ✅ Complete | Year‑over‑year change in total assets (∆AT/AT) computed from Compustat with a six‑month reporting lag【365329894510713†L44-L52】. |
| **Profitability** | `makeProfSignal` | ✅ Complete | Operating profitability (revenue minus COGS, SG&A and interest) divided by book equity plus minority interest【365329894510713†L44-L52】. Six‑month reporting lag applied. |

All signal functions live in `aa/signals/` and return tidy `DataFrame`s with columns `date`, `permno` and `signal`.  The timing conventions mirror those of Fama–French and Novy‑Marx.  Any deviations from the MATLAB code are noted in the respective docstrings.

## Sorting Routines

| Feature | MATLAB function | Status | Notes |
|-------|----------------|--------|------|
| **Univariate sorts** | `runUnivSort` | ✅ Complete (Milestone 1) | Supports arbitrary breakpoints, NYSE/conditional filters and produces high–low spreads. |
| **Double sorts** | `runBivSort` | ✅ Complete | Independent and conditional sorts implemented via `aa.asset_pricing.double_sort`. High–low spreads along both dimensions are returned. |
| **Characteristic‑managed portfolios** | `runCharPort` | ✅ Complete | Implemented in `aa.asset_pricing.characteristic.characteristic_managed_portfolio`. Computes cross‑sectional covariance between lagged characteristics and next‑month returns with optional standardisation. |

## Cross‑Sectional Regression

| Feature | MATLAB function | Status | Notes |
|-------|----------------|--------|------|
| **Fama–MacBeth regression** | `runFamaMacBeth` | ✅ Complete | Multi‑factor regression with Newey–West HAC standard errors. Returns average coefficients, standard errors, t‑statistics and counts of periods. |
| **Automatic factor alignment** | – | ✅ Complete | The regression engine aligns regressors and dependent variables by date and permno, dropping missing observations. |
| **Cluster/HAC options** | – | ✅ Complete | Lags may be set via the `nw_lags` argument.  Time clustering beyond Newey–West is approximated via the long‑run variance estimator. |

## Reporting

| Feature | MATLAB function | Status | Notes |
|-------|----------------|--------|------|
| **Portfolio return tables** | `printUnivSort`/`printBivSort` | ✅ Complete | `aa.reporting.tables.portfolio_returns_table` produces Markdown and LaTeX tables for 2D sorts. |
| **High–low spread tables** | – | ✅ Complete | `high_low_table` summarises high–low spreads, either time‑series or averages. |
| **Fama–MacBeth tables** | – | ✅ Complete | `fama_macbeth_table` formats regression coefficients, standard errors, t‑stats and observation counts. |
| **Alpha tables** | – | ✅ Pending | Extension for multi‑factor alpha and model fit metrics is deferred to a future milestone. |

## Tests

Unit tests are provided for each signal, double sorts, characteristic‑managed portfolios and Fama–MacBeth regressions under the `tests/` directory.  One integration test verifies that three anomalies can be combined to perform sorts and regressions without error.  Tolerances for small numerical differences are implicit in the use of synthetic data; future work should benchmark against MATLAB outputs using CRSP/Compustat data.

## Numerical Tolerances

This Python port uses double‑precision arithmetic throughout and relies on pandas/numpy operations.  Minor deviations from MATLAB results may occur due to differences in default handling of missing values and floating‑point rounding.  Extensive cross‑checks on CRSP/Compustat data are planned for future milestones.
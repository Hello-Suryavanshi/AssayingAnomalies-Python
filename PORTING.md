## Porting `use_library.m` to Python

The MATLAB script `use_library.m` demonstrates how to employ the Assaying
Anomalies toolkit by loading CRSP and Compustat data, constructing a simple
size signal, forming portfolios and running cross‑sectional regressions.
The Python pipeline implemented in this repository mirrors those steps
and surfaces them as composable functions.

### Mapping of steps

| MATLAB (`use_library.m`)               | Python equivalent                                   |
|---------------------------------------|------------------------------------------------------|
| Load CRSP monthly via `get_crsp_monthly`  | `aa.io.load_crsp` or user‑provided DataFrame        |
| Load Compustat FUNDA via `get_compustat_funda` | `aa.io.load_compustat`                      |
| Load CCM link history via `get_ccm_linkhist` | `aa.io.load_link`                         |
| Prepare Compustat (`prepare_compustat_annual`) | **Deferred** in this milestone; stub omitted |
| Link GVKEY to PERMNO (`map_gvkey_to_permno`) | **Deferred** in this milestone; stub omitted |
| Lag market equity and compute size signal | `aa.signals.compute_size_signal`                  |
| Form portfolios via `makeUnivSortInd`/`runUnivSort` | `aa.asset_pricing.univariate_sort`           |
| Compute H–L returns                       | Provided by `univariate_sort` summary and time series |
| Run Fama–MacBeth regressions (`runFamaMacBeth`) | `aa.asset_pricing.fama_macbeth`               |
| Display tables                            | `aa.reporting.portfolio_summary_md`, `regression_table` |

### Implemented vs deferred

* **Implemented**: Data loading helpers for CRSP, Compustat and CCM (though the
  latter two are currently pass‑throughs when supplied as DataFrames), size
  signal computation (lagged −log ME), univariate portfolio sorts (EW & VW),
  high‑minus‑low series, Fama–MacBeth regressions with Newey–West errors,
  Markdown table generation, and an end‑to‑end pipeline script.

* **Deferred**: Compustat cleaning and GVKEY‑to‑PERMNO linking routines,
  full support for additional anomaly signals, multi‑signal sorts,
  dynamic weighting schemes, integration with WRDS downloads, and advanced
  reporting/visualisation remain to be implemented in future milestones.
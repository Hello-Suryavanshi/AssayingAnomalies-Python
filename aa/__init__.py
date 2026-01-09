"""
Assaying Anomalies Python Package
================================

This package contains modules for constructing anomaly signals, forming
portfolios, running cross‑sectional regressions and producing
publication‑quality tables.  It mirrors the structure of the MATLAB
Assaying Anomalies toolkit by Novy‑Marx & Velikov while adhering to
Pythonic design principles:

* Functions are pure, with no global state or hidden side effects.
* Modules are fully typed and documented.
* Data flows through pandas ``DataFrame`` objects using tidy
  conventions: each row represents an observation and each column a
  variable.
* Pipelines are composable – the output of one stage feeds directly
  into the next.

Subpackages and modules
-----------------------

The top‑level ``aa`` package exposes the following submodules:

``signals``
    Construction of firm‑level anomaly signals (size, book‑to‑market,
    momentum, investment and profitability).  Each function accepts
    raw CRSP/Compustat data and returns a DataFrame of lagged
    characteristics.

``asset_pricing``
    Portfolio sorts, characteristic‑managed portfolios and
    cross‑sectional regressions.  Includes univariate and double
    portfolio sorts and Fama–MacBeth estimators.

``reporting``
    Utilities for producing Markdown and LaTeX tables suitable for
    inclusion in academic papers or reports.

In addition to the original modules, milestone 4 introduces several
new modules: ``aa.regime`` for regime analysis, ``aa.robustness`` for
systematic sensitivity checks, ``aa.diagnostics`` for stability
diagnostics, ``aa.multiple_testing`` for multiple‑testing controls, and
``aa.simulation`` for placebo experiments.  These modules are not
imported by default but can be accessed directly via ``aa.regime`` and
similarly for others.
"""

from . import asset_pricing, reporting  # noqa: F401  # re-export subpackages

__all__ = [
    "asset_pricing",
    "reporting",
]

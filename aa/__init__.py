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

Subpackages
-----------

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

``prep``
    Panel construction utilities for merging CRSP, Compustat and CCM
    data.  In this simplified port the Compustat cleaning routines
    remain deferred; however the interface is provided for future
    extension.
"""

from . import signals, asset_pricing, reporting

__all__ = ["signals", "asset_pricing", "reporting"]

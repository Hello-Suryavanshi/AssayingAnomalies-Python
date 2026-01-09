"""
Anomaly signal constructors
===========================

This subpackage contains functions for constructing a variety of
cross‑sectional anomaly signals from CRSP and Compustat data.  Each
function implements the timing conventions used by Fama–French and
Novy‑Marx et al.  Signals are returned as tidy pandas DataFrames with
columns ``date``, ``permno`` and ``signal``.  Missing inputs result
in missing (``NaN``) signals so that downstream code can handle
imputation or deletion explicitly.

The currently supported signals are:

* :func:`compute_size_signal` – negative logarithm of lagged market
  equity (size anomaly).
* :func:`compute_book_to_market_signal` – log book‑to‑market ratio
  based on lagged book equity and market equity.
* :func:`compute_momentum_signal` – 12‑minus‑2 momentum using
  cumulative returns from month *t−12* through *t−2*.
* :func:`compute_investment_signal` – investment ratio using the
  year‑over‑year change in total assets.
* :func:`compute_profitability_signal` – operating profitability
  divided by book equity plus minority interest.

Each constructor resides in its own module to keep dependencies
minimal.  Refer to the respective module docstrings for detailed
references and formulae.
"""

from .size import compute_size_signal
from .book_to_market import compute_book_to_market_signal
from .momentum import compute_momentum_signal
from .investment import compute_investment_signal
from .profitability import compute_profitability_signal

__all__ = [
    "compute_size_signal",
    "compute_book_to_market_signal",
    "compute_momentum_signal",
    "compute_investment_signal",
    "compute_profitability_signal",
]

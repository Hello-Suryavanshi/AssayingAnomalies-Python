"""
Asset pricing routines (portfolio sorts, regressions).

This subpackage exposes functions for constructing portfolios and
estimating risk prices from characteristic data.  It includes

* Univariate sorts (:func:`univariate_sort`) and their configuration
  via :class:`aa.asset_pricing.univariate.SortConfig`.
* Two‑dimensional sorts (:func:`double_sort`) with both independent
  and conditional breakpoints governed by
  :class:`aa.asset_pricing.double_sort.DoubleSortConfig`.
* Characteristic‑managed portfolios (:func:`characteristic_managed_portfolio`).
* Cross‑sectional regression estimators (Fama–MacBeth) returning
  coefficients, standard errors, t‑statistics and observation counts.

Importing this module brings the most commonly used routines into the
``aa.asset_pricing`` namespace.
"""

from .univariate import SortConfig, univariate_sort  # noqa: F401
from .double_sort import DoubleSortConfig, double_sort  # noqa: F401
from .fama_macbeth import fama_macbeth, fama_macbeth_full  # noqa: F401

__all__ = [
    "SortConfig",
    "univariate_sort",
    "DoubleSortConfig",
    "double_sort",
    "fama_macbeth",
    "fama_macbeth_full",
]

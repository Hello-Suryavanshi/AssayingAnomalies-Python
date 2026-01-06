"""
Asset pricing routines (portfolio sorts, factor models, regressions).
"""

from .fama_macbeth import fama_macbeth
from .univariate import SortConfig, univariate_sort

__all__ = ["fama_macbeth", "SortConfig", "univariate_sort"]

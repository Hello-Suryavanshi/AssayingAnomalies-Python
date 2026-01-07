"""Data cleaning and linking routines.

This subpackage exposes the high‑level panel building routine used by the
pipeline. It deliberately avoids importing the original MATLAB compat
functions (`prepare_compustat_annual` and `map_gvkey_to_permno`) because
those modules are not included in this simplified port. If you need
additional prep functionality, import the relevant functions directly
from the submodules.
"""

# Expose build_monthly_panel for convenience
from .build_panel import build_monthly_panel  # noqa: F401

__all__ = ["build_monthly_panel"]

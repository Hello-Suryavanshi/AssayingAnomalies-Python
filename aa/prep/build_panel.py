"""
Panel construction utilities.

This module defines functions to merge raw CRSP returns, Compustat
fundamental data, and CCM link history into a unified monthly panel.
The resulting panel can then be used to construct anomaly signals and
run asset‑pricing tests.  For the size anomaly considered in
``run_size_pipeline``, the Compustat and link data are not required,
but stub arguments are accepted to support future extensions.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# NOTE: The Compustat and linktable helpers are deliberately omitted from
# this simplified port. The original MATLAB toolkit includes routines
# to clean Compustat annual data and link GVKEYs to PERMNOs.  Because
# the size anomaly does not require fundamental data, these functions
# are not implemented here.  When both ``funda`` and ``lnkhist`` are
# provided to :func:`build_monthly_panel`, the data are ignored with a
# warning so the caller is aware of the limitation.
import warnings


def build_monthly_panel(
    *,
    crsp: pd.DataFrame,
    funda: Optional[pd.DataFrame] = None,
    lnkhist: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble a monthly asset panel from CRSP, Compustat and CCM link data.

    Parameters
    ----------
    crsp : DataFrame
        Monthly CRSP data with at minimum ``date``, ``permno``, ``ret``,
        and ``me`` (market equity).
    funda : DataFrame, optional
        Raw Compustat FUNDA table; if provided, will be cleaned and linked
        to PERMNO via the CCM link history.  Rows where a valid link
        exists will be merged on assignment month.
    lnkhist : DataFrame, optional
        CCM link history table used to map GVKEYs to PERMNOs.  Required
        if ``funda`` is provided.

    Returns
    -------
    DataFrame
        A panel with CRSP returns and, if available, matched Compustat
        fundamentals.  For periods with no match, Compustat columns are
        NaN.  The returned DataFrame is sorted by ``permno`` and ``date``.
    """
    # Start with CRSP copy to avoid mutating caller data
    panel = crsp.copy()

    # If we have Compustat + link data, we would normally clean and merge.
    # However, this simplified port does not implement those routines.  Warn
    # the user that the provided fundamental data will be ignored.
    if funda is not None or lnkhist is not None:
        warnings.warn(
            "Compustat and link data are ignored in this simplified implementation.",
            RuntimeWarning,
        )

    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    return panel

"""
Multiple testing and data‑snooping corrections.

When evaluating a panel of anomalies or factors, it is important to
control the probability of false discoveries arising simply by
chance.  This module implements classic corrections including the
Bonferroni adjustment for controlling the family‑wise error rate
(FWER) and the Benjamini–Hochberg (BH) procedure for controlling the
false discovery rate (FDR).  These procedures are described in
standard statistics texts and lectures; see for example the
definition of the BH procedure at【239467018970311†L594-L606】.

Functions
---------
bonferroni_correction(p_values, alpha)
    Adjust p‑values and significance threshold using the Bonferroni
    method.  Returns adjusted p‑values and the cut‑off level.

benjamini_hochberg(p_values, q)
    Apply the BH procedure to a set of p‑values and return a
    boolean mask indicating which hypotheses are rejected.

adjust_pvalues(p_values, method)
    Compute adjusted p‑values according to the specified method
    ('bonferroni' or 'BH').

fdr_table(p_values, methods)
    Construct a DataFrame summarising raw and adjusted p‑values and
    rejection decisions for multiple methods.

"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "bonferroni_correction",
    "benjamini_hochberg",
    "adjust_pvalues",
    "fdr_table",
]


def bonferroni_correction(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, float]:
    """Apply the Bonferroni correction to a sequence of p‑values.

    The Bonferroni method controls the probability of making one or
    more Type I errors (family‑wise error rate) by testing each
    hypothesis at level ``alpha / m``, where ``m`` is the number of
    hypotheses.  Equivalently, the adjusted p‑values are ``p * m``
    capped at 1.

    Parameters
    ----------
    p_values : sequence of float
        Raw p‑values from individual tests.
    alpha : float, default 0.05
        Desired family‑wise significance level.

    Returns
    -------
    adjusted_pvalues : ndarray
        Adjusted p‑values.  Each is ``min(p * m, 1)``.
    cutoff : float
        Bonferroni significance threshold equal to ``alpha / m``.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    if m == 0:
        return p, alpha
    adj = np.minimum(p * m, 1.0)
    cutoff = alpha / float(m)
    return adj, cutoff


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Apply the Benjamini–Hochberg procedure to control the FDR.

    The BH procedure sorts the p‑values in ascending order, finds the
    largest index ``r`` such that ``p_(r) ≤ (r / m) q`` (where ``m`` is
    the number of tests), and rejects all hypotheses with p‑value
    less than or equal to ``p_(r)``【239467018970311†L594-L606】.  When no p‑value
    satisfies the criterion, no hypotheses are rejected.

    Parameters
    ----------
    p_values : sequence of float
        Raw p‑values.
    q : float, default 0.05
        Desired false discovery rate.

    Returns
    -------
    ndarray
        Boolean array of the same length as ``p_values`` indicating
        which hypotheses are rejected.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    # Sort p‑values and keep original order
    order = np.argsort(p)
    p_sorted = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    below = p_sorted <= thresholds
    if not np.any(below):
        # No rejections
        return np.zeros(m, dtype=bool)
    # Largest r satisfying criterion
    r = np.max(np.where(below)[0])
    cutoff = p_sorted[r]
    # Reject all p‑values ≤ cutoff
    rejects = p <= cutoff
    return rejects


def adjust_pvalues(p_values: Sequence[float], method: str = "bonferroni") -> np.ndarray:
    """Compute adjusted p‑values under a multiple testing method.

    Parameters
    ----------
    p_values : sequence of float
        Raw p‑values.
    method : {'bonferroni', 'BH'}, default 'bonferroni'
        Correction method.  ``'bonferroni'`` multiplies p‑values by
        ``m`` and caps at 1; ``'BH'`` returns the BH step‑up values
        defined as ``p_(i) m / i`` for the sorted p‑values.

    Returns
    -------
    ndarray
        Adjusted p‑values.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    method = method.lower()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    elif method == "bh" or method == "benjamini‑hochberg":
        # BH adjusted p‑values
        order = np.argsort(p)
        p_sorted = p[order]
        adj = np.zeros(m, dtype=float)
        # Compute p_(i) * m / i and ensure monotonicity
        bh_vals = p_sorted * m / (np.arange(1, m + 1))
        bh_adj = np.minimum.accumulate(bh_vals[::-1])[::-1]
        # Map back to original order
        adj[order] = np.minimum(bh_adj, 1.0)
        return adj
    else:
        raise ValueError("method must be 'bonferroni' or 'BH'")


def fdr_table(
    p_values: Sequence[float],
    methods: Iterable[str] = ("bonferroni", "BH"),
    alpha: float = 0.05,
    q: float = 0.05,
) -> pd.DataFrame:
    """Create a table summarising multiple testing adjustments.

    Parameters
    ----------
    p_values : sequence of float
        Raw p‑values.
    methods : iterable of str, default ('bonferroni', 'BH')
        Correction methods to apply.  Supported values are
        ``'bonferroni'`` and ``'BH'`` (alias for
        Benjamini–Hochberg).  Additional values are ignored.
    alpha : float, default 0.05
        Significance level for the Bonferroni correction.
    q : float, default 0.05
        Target FDR for the BH procedure.

    Returns
    -------
    DataFrame
        Columns include the raw p‑values and adjusted p‑values for each
        specified method, as well as boolean rejection decisions where
        applicable.
    """
    p = np.asarray(p_values, dtype=float)
    p_list = p.tolist()

    df = pd.DataFrame({"p": p})

    adj_bonf, cutoff = bonferroni_correction(p_list, alpha)
    df["p_adj_bonf"] = adj_bonf
    df["reject_bonf"] = p <= cutoff

    adj_bh = adjust_pvalues(p_list, method="BH")
    rejects_bh = benjamini_hochberg(p_list, q)
    df["p_adj_bh"] = adj_bh
    df["reject_bh"] = rejects_bh

    return df

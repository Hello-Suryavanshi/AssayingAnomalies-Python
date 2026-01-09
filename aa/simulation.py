"""
Simulation and placebo framework for anomaly evaluation.

Empirical researchers often wish to benchmark the observed performance
of an anomaly against what could be expected under random noise.  This
module provides simple utilities to generate random or permuted
signals, perform permutation and bootstrap tests, and construct
empirical null distributions for portfolio sorts and regressions.

The emphasis is on transparency: random signals are generated
independently of the observed returns to avoid inadvertent look‑ahead
bias, and permutation tests preserve the joint distribution of
returns by shuffling the characteristic across assets or time.  When
integrated with the rest of the library, these tools allow users to
visualise the distribution of t‑statistics or high–low spreads under
the null and compare them with the realised anomaly.

"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "random_signal_like",
    "permutation_test",
    "bootstrap_placebo",
    "simulate_null_distribution",
]


def random_signal_like(
    signal_df: pd.DataFrame,
    *,
    method: str = "gaussian",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a random signal DataFrame with the same index as ``signal_df``.

    Parameters
    ----------
    signal_df : DataFrame
        Existing signal with columns at least ``'date'``, ``'permno'`` and
        ``'signal'``.  The returned DataFrame will copy the first two
        columns and replace ``'signal'`` with random values.
    method : {'gaussian', 'permute', 'bootstrap'}, default 'gaussian'
        Method for generating random values.  ``'gaussian'`` draws
        independent standard normal values; ``'permute'`` randomly
        permutes the existing ``signal`` column; ``'bootstrap'`` draws
        values with replacement from the existing signal.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        A copy of ``signal_df`` with the ``'signal'`` column replaced by
        random or permuted values.
    """
    rng = np.random.default_rng(seed)
    out = signal_df[[col for col in signal_df.columns if col != "signal"]].copy()
    n = len(signal_df)
    if method == "gaussian":
        rand_vals = rng.standard_normal(n)
    elif method == "permute":
        rand_vals = np.asarray(signal_df["signal"], dtype=float).copy()
        rng.shuffle(rand_vals)
    elif method == "bootstrap":
        orig = np.asarray(signal_df["signal"], dtype=float)
        rand_vals = rng.choice(orig, size=n, replace=True)
    else:
        raise ValueError("method must be 'gaussian', 'permute' or 'bootstrap'")
    out["signal"] = rand_vals
    return out


def permutation_test(
    *,
    analysis_fn: Callable[..., Any],
    n_iter: int,
    metric_fn: Callable[[Any], float],
    signal_df_key: str = "signal",
    seed: Optional[int] = None,
    **data_kwargs: Any,
) -> pd.Series:
    """Perform a permutation test by shuffling the signal and recomputing the metric.

    Parameters
    ----------
    analysis_fn : callable
        Analysis function to be applied to the data.  Must accept the
        same keyword arguments as those passed in ``data_kwargs``.  The
        ``signal`` argument will be replaced by its permuted version in
        each iteration.
    n_iter : int
        Number of permutations to perform.
    metric_fn : callable
        Function that extracts a scalar metric from the analysis result
        (e.g. a t‑statistic or high–low return).  The permutation test
        returns the distribution of this metric under random labelling.
    signal_df_key : str, default 'signal'
        Name of the keyword argument corresponding to the signal DataFrame.
    seed : int or None, optional
        Random seed for reproducibility.
    **data_kwargs : DataFrame or other
        Arguments forwarded to ``analysis_fn``.  At a minimum must
        include the entry keyed by ``signal_df_key``.

    Returns
    -------
    Series
        A Series of length ``n_iter`` containing the metric computed on
        each permuted dataset.  The index corresponds to the iteration.
    """
    if signal_df_key not in data_kwargs:
        raise KeyError(f"data_kwargs must contain a '{signal_df_key}' argument")
    original_signal = data_kwargs[signal_df_key]
    rng = np.random.default_rng(seed)
    metrics: List[float] = []
    for i in range(n_iter):
        # Permute signal
        permuted = random_signal_like(
            original_signal, method="permute", seed=int(rng.integers(0, 2**31 - 1))
        )
        data_kwargs_perm = dict(data_kwargs)
        data_kwargs_perm[signal_df_key] = permuted
        try:
            res = analysis_fn(**data_kwargs_perm)
        except Exception as exc:
            raise RuntimeError(
                f"analysis_fn failed during permutation iteration {i}: {exc}"
            )
        metric = metric_fn(res)
        metrics.append(float(metric))
    return pd.Series(metrics)


def bootstrap_placebo(
    *,
    analysis_fn: Callable[..., Any],
    n_iter: int,
    metric_fn: Callable[[Any], float],
    signal_df_key: str = "signal",
    seed: Optional[int] = None,
    **data_kwargs: Any,
) -> pd.Series:
    """Perform a bootstrap placebo experiment by resampling signals with replacement.

    Parameters
    ----------
    analysis_fn, n_iter, metric_fn, signal_df_key, seed, data_kwargs
        As in :func:`permutation_test`, except that the signal is drawn
        with replacement from its empirical distribution rather than
        permuted.

    Returns
    -------
    Series
        Distribution of the metric under the bootstrap placebo.
    """
    if signal_df_key not in data_kwargs:
        raise KeyError(f"data_kwargs must contain a '{signal_df_key}' argument")
    original_signal = data_kwargs[signal_df_key]
    rng = np.random.default_rng(seed)
    metrics: List[float] = []
    for i in range(n_iter):
        boot = random_signal_like(
            original_signal, method="bootstrap", seed=int(rng.integers(0, 2**31 - 1))
        )
        data_kwargs_boot = dict(data_kwargs)
        data_kwargs_boot[signal_df_key] = boot
        try:
            res = analysis_fn(**data_kwargs_boot)
        except Exception as exc:
            raise RuntimeError(
                f"analysis_fn failed during bootstrap iteration {i}: {exc}"
            )
        metric = metric_fn(res)
        metrics.append(float(metric))
    return pd.Series(metrics)


def simulate_null_distribution(
    *,
    analysis_fn: Callable[..., Any],
    n_iter: int,
    metric_fn: Callable[[Any], float],
    signal_df_key: str = "signal",
    method: str = "gaussian",
    seed: Optional[int] = None,
    **data_kwargs: Any,
) -> pd.Series:
    """Generate an empirical null distribution via random signals.

    Parameters
    ----------
    analysis_fn, n_iter, metric_fn, signal_df_key, seed, data_kwargs
        As in :func:`permutation_test`, but the signal in each
        iteration is replaced by a synthetic one drawn from the method
        specified by ``method``.  This allows benchmarking the observed
        anomaly against signals with no information about returns.
    method : {'gaussian', 'bootstrap'}, default 'gaussian'
        Generation mechanism for random signals.  ``'gaussian'`` draws
        independent standard normal values; ``'bootstrap'`` samples with
        replacement from the empirical distribution of the original
        signal.

    Returns
    -------
    Series
        A Series of length ``n_iter`` containing the metric for each
        simulated signal.
    """
    if signal_df_key not in data_kwargs:
        raise KeyError(f"data_kwargs must contain a '{signal_df_key}' argument")
    original_signal = data_kwargs[signal_df_key]
    if method not in {"gaussian", "bootstrap"}:
        raise ValueError("method must be 'gaussian' or 'bootstrap'")
    rng = np.random.default_rng(seed)
    metrics: List[float] = []
    for i in range(n_iter):
        synthetic = random_signal_like(
            original_signal,
            method=method,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        data_kwargs_sim = dict(data_kwargs)
        data_kwargs_sim[signal_df_key] = synthetic
        try:
            res = analysis_fn(**data_kwargs_sim)
        except Exception as exc:
            raise RuntimeError(
                f"analysis_fn failed during simulation iteration {i}: {exc}"
            )
        metric = metric_fn(res)
        metrics.append(float(metric))
    return pd.Series(metrics)

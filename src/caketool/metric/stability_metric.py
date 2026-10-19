from typing import Literal

import numpy as np


def psi(
    expected: np.ndarray, actual: np.ndarray, bucket_type: Literal["bins", "quantiles"] = "quantiles", n_bins: int = 10
) -> float:
    """
    Calculate the Population Stability Index (PSI) for two arrays.

    Parameters
    ----------
    expected : np.ndarray
        Array of expected values.
    actual : np.ndarray
        Array of actual values.
    bucket_type : Literal["bins", "quantiles"], optional
        Binning strategy. Accepts two options: 'bins' and 'quantiles'. Defaults to 'quantiles'.
        'bins': input arrays are split into bins with equal and fixed steps based on 'expected' array.
        'quantiles': input arrays are binned according to 'expected' array with given number of n_bins.
    n_bins : int, optional
        Number of buckets for binning. Defaults to 10.

    Returns
    -------
    float
        A single float number representing the PSI value.

    Notes
    -----
    - The outermost bin edges are ``-inf`` and ``+inf`` so values of ``actual``
      outside the range of ``expected`` are still counted.
    - Duplicate quantile edges (e.g. zero-inflated features) are merged.
    - Missing values (NaN) are counted in a separate bin.
    """
    if bucket_type not in ("bins", "quantiles"):
        raise ValueError(f"bucket_type must be 'bins' or 'quantiles'. Got: {bucket_type!r}")

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected_valid = expected[~np.isnan(expected)]
    actual_valid = actual[~np.isnan(actual)]

    if len(expected_valid) == 0:
        bins = np.array([-np.inf, np.inf])
    elif bucket_type == "bins":
        bins = np.linspace(expected_valid.min(), expected_valid.max(), n_bins + 1)
    else:
        percentage = np.arange(0, n_bins + 1) / n_bins * 100
        bins = np.percentile(expected_valid, percentage)

    # Merge duplicate edges and open the outer edges so out-of-range values are counted
    bins = np.unique(bins)
    bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

    # Calculate frequencies, with missing values in their own bin
    expected_counts = np.append(np.histogram(expected_valid, bins)[0], len(expected) - len(expected_valid))
    actual_counts = np.append(np.histogram(actual_valid, bins)[0], len(actual) - len(actual_valid))

    return psi_from_distribution(expected_counts / len(expected), actual_counts / len(actual))


def psi_from_distribution(expected_percents: np.ndarray, actual_percents: np.ndarray, eps: float = 1e-5) -> float:
    """
    Calculate PSI from distributions of expected and actual percentages.

    Parameters
    ----------
    expected_percents : np.ndarray
        Array of expected percentages.
    actual_percents : np.ndarray
        Array of actual percentages.
    eps : float, optional
        Lower bound applied to every bin share to avoid ``log(0)`` and division
        by zero. Defaults to ``1e-5``.

    Returns
    -------
    float
        A single float number representing the PSI value.
    """
    # Normalize to sum to 1
    expected_percents = np.asarray(expected_percents, dtype=float)
    actual_percents = np.asarray(actual_percents, dtype=float)
    expected_percents = expected_percents / np.sum(expected_percents)
    actual_percents = actual_percents / np.sum(actual_percents)

    # Clip frequencies to avoid division by zero
    expected_percents = np.clip(expected_percents, a_min=eps, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=eps, a_max=None)

    # Calculate PSI
    psi_value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return float(np.sum(psi_value))

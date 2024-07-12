from typing import Literal
import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bucket_type: Literal["bins", "quantiles"] = "quantiles", n_bins: int = 10) -> float:
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
    """
    if bucket_type == "bins":
        min_val = expected.min()
        max_val = expected.max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
    elif bucket_type == "quantiles":
        percentage = np.arange(0, n_bins + 1) / n_bins * 100
        bins = np.percentile(expected, percentage)

    # Calculate frequencies
    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)

    return psi_from_distribution(expected_percents, actual_percents)


def psi_from_distribution(expected_percents: np.ndarray, actual_percents: np.ndarray) -> float:
    """
    Calculate PSI from distributions of expected and actual percentages.

    Parameters
    ----------
    expected_percents : np.ndarray
        Array of expected percentages.
    actual_percents : np.ndarray
        Array of actual percentages.

    Returns
    -------
    float
        A single float number representing the PSI value.
    """
    # Normalize to sum to 1
    expected_percents = expected_percents / np.sum(expected_percents)
    actual_percents = actual_percents / np.sum(actual_percents)

    # Clip frequencies to avoid division by zero
    expected_percents = np.clip(expected_percents, a_min=0.00001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.00001, a_max=None)

    # Calculate PSI
    psi_value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_value = np.sum(psi_value)

    return psi_value

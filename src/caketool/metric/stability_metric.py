from typing import Literal
import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bucket_type: Literal["bins", "quantiles"] = "quantiles", n_bins: int = 10) -> float:
    """Calculate PSI metric for two arrays.

    Parameters
    ----------
        expected : list-like
            Array of expected values
        actual : list-like
            Array of actual values
        bucket_type : str
            Binning strategy. Accepts two options: 'bins' and 'quantiles'. Defaults to 'bins'.
            'bins': input arrays are splitted into bins with equal
                and fixed steps b
                ased on 'expected' array
            'quantiles': input arrays are binned according to 'expected' array
                with given number of n_bins
        n_bins : int
            Number of buckets for binning. Defaults to 10.

    Returns
    -------
        A single float number
    """
    if bucket_type == "bins":
        min_val = expected.min()
        max_val = expected.max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
    elif bucket_type == "quantiles":
        percentage = np.arange(0, n_bins + 1) / (n_bins) * 100
        bins = np.percentile(expected, percentage)
    # Calculate frequencies
    expected_percents = np.histogram(expected, bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins)[0] / len(actual)
    # Clip freaquencies to avoid zero division
    expected_percents = np.clip(expected_percents, a_min=0.00001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.00001, a_max=None)
    # Calculate PSI
    psi_value = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_value = sum(psi_value)
    return psi_value

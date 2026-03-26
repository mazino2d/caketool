import numpy as np


def create_percentile_bins(data: np.ndarray, n: int) -> list[float]:
    """Compute *n* equal-frequency bin edges from *data* using percentiles.

    Divides the data range into *n* buckets such that each bucket contains
    approximately the same number of observations (quantile binning).

    Parameters
    ----------
    data : np.ndarray
        1-D array of numeric values to bin.
    n : int
        Number of bins.  The function returns ``n + 1`` edge values,
        from the minimum (0th percentile) to the maximum (100th percentile).

    Returns
    -------
    list[float]
        Sorted list of ``n + 1`` bin edge values.

    Examples
    --------
    >>> import numpy as np
    >>> create_percentile_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), n=4)
    [1.0, 3.25, 5.5, 7.75, 10.0]
    """
    percentiles = np.linspace(0, 100, n + 1)
    bin_edges = np.percentile(data, percentiles)
    return list(bin_edges)

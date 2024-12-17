from typing import List
import numpy as np

def create_percentile_bins(data: np.ndarray, n: int) -> List[float]:
    percentiles = np.linspace(0, 100, n + 1)
    bin_edges = np.percentile(data, percentiles)
    return list(bin_edges)

import numpy as np

def create_percentile_bins(data, n):
    percentiles = np.linspace(0, 100, n + 1)
    bin_edges = np.percentile(data, percentiles)
    return bin_edges

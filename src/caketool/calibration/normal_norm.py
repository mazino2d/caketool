import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from typing import Union, List, Tuple

def calibrate_score_to_normal(scores: Union[List, Tuple, np.ndarray, pd.Series, float], standard=False) -> Union[np.ndarray, float]:
    """
    Calibrates scores to a normal distribution using the probit (inverse normal CDF) transformation, 
    and optionally standardizes the result.

    Parameters:
    ----------
    scores : Union[List, Tuple, np.ndarray, pd.Series, float]
        Input scores in the range [0, 1], provided as a list, tuple, numpy array, pandas Series, 
        or a single float.
        
    standard : bool, optional, default=False
        If True, standardizes the z-scores (subtracts mean, divides by std).

    Returns:
    -------
    calibrated_score : Union[np.ndarray, float]
        Calibrated scores in the range (0, 1), returned as a numpy array or float depending on input type.
    """
    # Clip scores to avoid issues with extreme values (0 and 1)
    scores = np.clip(scores, 1e-9, 1 - 1e-9)
    
    # Convert to z-scores using inverse normal CDF
    z_scores = norm.ppf(scores)
    
    # Standardize if required
    if standard:
        z_scores = (z_scores - np.mean(z_scores)) / np.std(z_scores)
    
    # Apply sigmoid to map back to (0, 1)
    calibrated_score = expit(z_scores)
    
    return calibrated_score
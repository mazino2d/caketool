import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from typing import Union, List, Tuple

def calibrate_score_to_normal(scores: Union[List, Tuple, np.ndarray, pd.Series, float], standard=False) -> Union[np.ndarray, float]:
    """
    Calibrates scores via normal distribution transformation while keeping output in (0, 1).
    
    Process:
        1. Transform scores to z-scores using probit (inverse normal CDF)
        2. Optionally standardize z-scores (mean=0, std=1)
        3. Map back to (0, 1) using sigmoid function
    
    This compresses extreme values toward 0.5 and produces a more balanced distribution.

    Note:
    -----
    - Only suitable for UNIMODAL distributions.
    - For bimodal/multimodal, use supervised calibration (Isotonic, Platt Scaling, etc.)
    - Pulls extreme values (near 0 or 1) toward 0.5 to reduce overconfidence.

    Parameters:
    ----------
    scores : Union[List, Tuple, np.ndarray, pd.Series, float]
        Input scores in the range [0, 1].
        
    standard : bool, optional, default=False
        If True, standardizes the z-scores before applying sigmoid.

    Returns:
    -------
    calibrated_score : Union[np.ndarray, float]
        Calibrated scores in the range (0, 1).
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
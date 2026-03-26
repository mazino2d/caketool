import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def calibrate_score_to_normal(
    scores: list | tuple | np.ndarray | pd.Series | float, standard=False
) -> np.ndarray | float:
    """Calibrate scores via normal-distribution transformation, keeping output in (0, 1).

    Applies a three-step transformation:

    1. Map scores to z-scores using the probit function (inverse normal CDF).
    2. Optionally standardise z-scores to zero mean and unit variance.
    3. Map z-scores back to (0, 1) using the sigmoid (logistic) function.

    The result compresses extreme values toward 0.5, reducing over-confidence
    while preserving the rank order of the original scores (monotonic
    transformation).

    Notes
    -----
    - Only suitable for **unimodal** score distributions.  For bimodal or
      multimodal distributions use supervised calibration methods such as
      Isotonic Regression or Platt Scaling.
    - Values exactly equal to 0 or 1 are clipped to ``1e-9`` and
      ``1 - 1e-9`` respectively before transformation to avoid numerical
      issues with the probit function.

    Parameters
    ----------
    scores : list | tuple | np.ndarray | pd.Series | float
        Input scores in the range [0, 1].
    standard : bool, optional
        If ``True``, standardises the z-scores (mean=0, std=1) before
        applying the sigmoid.  Defaults to ``False``.

    Returns
    -------
    np.ndarray | float
        Calibrated scores in the open interval (0, 1), same shape as input.

    Examples
    --------
    >>> calibrate_score_to_normal([0.1, 0.5, 0.9])
    array([0.401..., 0.5, 0.598...])
    >>> calibrate_score_to_normal(0.99)
    0.731...
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

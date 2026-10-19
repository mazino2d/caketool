import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-9


def _to_z(scores) -> np.ndarray:
    """Map scores in [0, 1] to z-scores with the probit function."""
    # Clip scores to avoid issues with extreme values (0 and 1)
    return norm.ppf(np.clip(scores, _EPS, 1 - _EPS))


def _standardize(z_scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize z-scores, only centering them when ``std`` is zero."""
    if std > 0:
        return (z_scores - mean) / std
    return z_scores - mean


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
        applying the sigmoid.  Defaults to ``False``.  The mean and std come
        from ``scores`` itself, so each output depends on the rest of the
        batch; for online or per-row scoring, use ``NormalScoreCalibrator``,
        which learns them from a reference set.  Constant scores (std = 0)
        are only centered, so they map to 0.5.

    Returns
    -------
    np.ndarray | float
        Calibrated scores in the open interval (0, 1), same shape as input.

    Examples
    --------
    >>> calibrate_score_to_normal([0.1, 0.5, 0.9])
    array([0.217..., 0.5, 0.782...])
    >>> calibrate_score_to_normal(0.99)
    0.911...
    """
    # Convert to z-scores using inverse normal CDF
    z_scores = _to_z(scores)

    # Standardize if required
    if standard:
        z_scores = _standardize(z_scores, float(np.mean(z_scores)), float(np.std(z_scores)))

    # Apply sigmoid to map back to (0, 1)
    calibrated_score = expit(z_scores)

    return calibrated_score


class NormalScoreCalibrator(TransformerMixin, BaseEstimator):
    """Normal-distribution calibration with z-score statistics learned from a reference set.

    Same transformation as ``calibrate_score_to_normal(..., standard=True)``,
    but the mean and std of the z-scores are computed once in ``fit`` (e.g. on
    the training or validation scores) and reused in ``transform``.  Each
    calibrated score therefore depends only on its own raw score, which makes
    it safe for online and per-row scoring.

    Attributes
    ----------
    mean_ : float
        Mean of the reference z-scores.
    std_ : float
        Standard deviation of the reference z-scores.  When it is zero,
        ``transform`` only centers the z-scores.

    Examples
    --------
    >>> calibrator = NormalScoreCalibrator().fit([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> calibrator.transform([0.5])
    array([0.5])
    """

    def fit(self, scores, y=None) -> "NormalScoreCalibrator":
        """Learn the z-score mean and std from reference scores in [0, 1].

        Parameters
        ----------
        scores : array-like
            Reference scores.
        y : None
            Ignored.

        Returns
        -------
        self
        """
        z_scores = _to_z(np.ravel(np.asarray(scores, dtype=float)))
        self.mean_ = float(np.mean(z_scores))
        self.std_ = float(np.std(z_scores))
        return self

    def transform(self, scores) -> np.ndarray | float:
        """Calibrate scores using the statistics learned in ``fit``.

        Parameters
        ----------
        scores : array-like or float
            Scores in [0, 1].

        Returns
        -------
        np.ndarray | float
            Calibrated scores in (0, 1), same shape as input.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError("NormalScoreCalibrator has not been fitted yet. Call .fit(scores) first.")
        return expit(_standardize(_to_z(scores), self.mean_, self.std_))

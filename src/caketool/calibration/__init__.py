"""Score calibration utilities.

Transforms raw model output scores into better-calibrated probabilities.
Currently supports normal-distribution-based calibration for unimodal score
distributions, either as a stateless function (``calibrate_score_to_normal``)
or as a fit/transform estimator (``NormalScoreCalibrator``).
"""

from .normal_norm import NormalScoreCalibrator as NormalScoreCalibrator
from .normal_norm import calibrate_score_to_normal as calibrate_score_to_normal

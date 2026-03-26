"""Score calibration utilities.

Transforms raw model output scores into better-calibrated probabilities.
Currently supports normal-distribution-based calibration for unimodal score
distributions.
"""

from .normal_norm import calibrate_score_to_normal as calibrate_score_to_normal

"""Risk reporting utilities for credit scoring models.

Provides:

- ``decribe_risk_score`` – Segments predicted probabilities into bands and
  computes per-band default rates, approval rates, and cumulative good/bad
  distributions for use in scorecard validation and monitoring reports.
"""

from .risk_report import decribe_risk_score as decribe_risk_score

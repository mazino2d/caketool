"""Risk reporting utilities for credit scoring models.

Provides:

- ``describe_risk_score`` – Segments predicted probabilities into bands and
  computes per-band default rates, approval rates, and cumulative good/bad
  distributions for use in scorecard validation and monitoring reports.
"""

from .risk_report import describe_risk_score as describe_risk_score

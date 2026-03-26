"""Model building utilities for binary classification.

Provides:

- ``BoostTree`` – XGBoost-based binary classifier with a built-in
  preprocessing pipeline (categorical encoding → infinity handling →
  univariate feature selection → collinear feature removal → XGBoost).
  Supports out-of-fold cross-validation via ``BoostTree.fit_oof``.
- ``EnsembleBoostTree`` – Ensemble of ``BoostTree`` models that averages
  predictions; typically used with OOF models.
- ``VotingModel`` – Generic ensemble that averages predictions from any
  list of scikit-learn-compatible estimators.
"""

from .boost_tree import BoostTree as BoostTree
from .boost_tree import EnsembleBoostTree as EnsembleBoostTree
from .voting_model import VotingModel as VotingModel

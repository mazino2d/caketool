"""Model explainability utilities using SHAP.

Provides a model-agnostic interface for computing global and local
feature attributions based on SHAP (SHapley Additive exPlanations).

Classes:

- ``ModelExplainer`` – Abstract base class defining the explainability
  interface (``fit``, ``get_feature_importance``, ``get_local_explanation``).
- ``PermutationExplainer`` – Concrete implementation using
  ``shap.PermutationExplainer``.  Works with any scikit-learn-compatible
  model without requiring access to model internals.

Example
-------
>>> from caketool.explainability import PermutationExplainer
>>> explainer = PermutationExplainer(model=fitted_model)
>>> explainer.fit(X_test)
>>> explainer.get_feature_importance()
>>> explainer.show_summary()
"""

from .base import ModelExplainer as ModelExplainer
from .permutation_explainer import PermutationExplainer as PermutationExplainer

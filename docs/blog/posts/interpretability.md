---
slug: interpretability
date: 2026-04-10
description: A model needs to be trusted to be used. The difference between interpretable and explainable models, SHAP vs LIME, and why consistency is the non-negotiable property of any explanation.
authors:
  - khoi
categories:
  - Modeling
tags:
  - interpretability
  - explainability
  - shap
---

# Interpretability & Explainability: Models Must Be Trusted

A model that no one understands will not be deployed. Or worse: it will be deployed and decisions made from it without anyone knowing when or why it's wrong. Interpretability is not a nicety — it is a condition for a model to be usable.

<!-- more -->

## Interpretable vs Explainable

These two terms are frequently conflated, but the distinction matters:

**Interpretable models** are inherently transparent. A human can trace through the model's logic for any individual prediction without external tools. Logistic regression, linear regression, decision trees (shallow), and scorecards fall into this category. The model itself *is* the explanation.

**Explainable models** are complex models paired with post-hoc explanation tools. The model (XGBoost, neural network) is not inherently transparent, but a separate tool approximates or decomposes its behavior. SHAP and LIME are the dominant tools in this category.

The choice between the two is not purely technical — it's shaped by:

- **Regulatory requirements**: some jurisdictions require explanations for individual decisions. Interpretable models provide them natively; explainable models require the explanation method to be legally defensible.
- **Stakeholder trust**: a decision tree a credit officer can follow themselves builds different trust than a SHAP waterfall plot they have to take on faith.
- **Model performance requirements**: if the performance gap between an interpretable and a complex model is large and the business stakes are high, explainability tools may be the right compromise.

## Global vs Local Explanations

**Global explanations** describe overall model behavior — what does the model rely on across all predictions?

- Feature importance (mean absolute SHAP values): ranks features by their average contribution to predictions across the dataset
- Partial Dependence Plots (PDP): shows the average effect of a single feature on the model's output, marginalizing over all other features

**Local explanations** describe individual predictions — why did the model predict X for this specific observation?

- SHAP waterfall plot: shows each feature's contribution to pushing the prediction above or below the baseline for one observation
- LIME: fits a local linear model around a specific observation to approximate the behavior of the complex model nearby

## SHAP: The Theory-Grounded Approach

SHAP (SHapley Additive exPlanations) computes each feature's contribution to a prediction based on Shapley values from cooperative game theory. Shapley values answer: "If each feature is a player in a game and the payout is the prediction, how much credit does each player deserve?"

SHAP has three key properties that make it the preferred choice:

**Efficiency**: the sum of all SHAP values equals the model's output minus the baseline (expected) output. Explanations are complete — they account for 100% of the prediction.

**Consistency**: if changing a model makes feature A more impactful on predictions, A's SHAP value will not decrease. This means SHAP values rank features in a way that is reliable across model changes.

**Accuracy**: the local explanation (sum of SHAP values) exactly equals the model's prediction. There's no approximation at the individual level.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global: feature importance
shap.summary_plot(shap_values, X_test)

# Local: single prediction breakdown
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist()
))
```

## LIME: Fast but Inconsistent

LIME (Local Interpretable Model-agnostic Explanations) works differently: it perturbs the input around a specific observation, generates a set of nearby predictions, and fits a simple linear model to those predictions. The linear model's coefficients are the "explanation."

LIME is faster than SHAP for complex models and works on any model type, including text and image models. But it has a fundamental weakness: the perturbation is random. Run LIME twice on the same observation and you may get different explanations. This inconsistency makes it unreliable for any context where the explanation itself carries weight — such as explaining a loan rejection to a customer or to a regulator.

Use LIME for quick exploratory analysis. For any explanation that will be acted on or audited, prefer SHAP.

## PDP and ICE

**Partial Dependence Plots (PDP)** show the average marginal effect of a feature on the target. They answer: "On average, how does the model's prediction change as I vary this feature, holding everything else constant?"

**Individual Conditional Expectation (ICE)** shows the same relationship for individual observations, rather than the average. Useful for detecting heterogeneous effects — when the relationship between a feature and the target is different for different subgroups. A flat PDP can hide crossing ICE curves, which means the average effect is misleading.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(model, X_train, features=["income", "credit_score"])
```

## Weight of Evidence and Scorecards in Credit Risk

In credit risk, interpretability standards are higher than in most ML domains. The industry standard tool is the **scorecard** — a linear model built on **Weight of Evidence (WoE)** transformed variables.

WoE for a bin = ln(% of goods in bin / % of bads in bin). It transforms each feature into a single number that directly measures its predictive power for the binary target. When all features are WoE-transformed, the logistic regression reduces to a simple addition of scores — a decision an analyst can compute by hand.

The Information Value (IV) derived from WoE measures each feature's overall discriminatory power:

- IV < 0.02: useless
- 0.02–0.1: weak predictor
- 0.1–0.3: medium predictor
- 0.3–0.5: strong predictor
- IV > 0.5: suspiciously strong — check for leakage

This framework sacrifices some predictive performance for complete transparency — every score point can be traced back to a specific feature and a specific bin. That traceability is what makes scorecards legally defensible.

## The Consistency Requirement

Regardless of the explanation method, one property is non-negotiable: **the same input must always produce the same explanation**.

A method that produces different explanations for the same observation on different runs (like LIME with random seeds) cannot be used for auditing, regulatory reporting, or customer-facing adverse action notices. Consistency is not a technical nicety — it's the property that makes an explanation trustworthy.

SHAP satisfies consistency by construction. If you need LIME for performance reasons, fix the random seed and document the dependency.

Interpretability is infrastructure. It enables model debugging, stakeholder trust, regulatory compliance, and responsible deployment. Building it in from the start — through model choice, explanation method selection, and consistency guarantees — costs far less than retrofitting it after a model is already in production.

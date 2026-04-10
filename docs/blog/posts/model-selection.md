---
slug: model-selection
date: 2026-04-10
description: No algorithm is universally best. The No Free Lunch Theorem, inductive bias, and the bias-variance tradeoff — choosing a model means choosing a set of assumptions.
authors:
  - khoi
categories:
  - Modeling
tags:
  - model-selection
  - algorithms
---

# Model Selection: No Free Lunch

Every algorithm embeds a set of assumptions about the structure of the data. Choosing a model is not about finding the "best" algorithm — it's about finding the algorithm whose assumptions best match your problem.

<!-- more -->

## The No Free Lunch Theorem

The No Free Lunch (NFL) Theorem states that no learning algorithm outperforms all others averaged across all possible problems. Any algorithm that performs well on some problems must necessarily perform worse on others.

The practical implication: **there is no universally best model**. Claims like "XGBoost always wins on tabular data" are heuristics based on historical contest results, not theorems. They're useful starting points, not conclusions.

What matters is choosing an algorithm whose inductive bias aligns with how your data was actually generated.

## Inductive Bias

Every learning algorithm makes assumptions that allow it to generalize from training data to unseen data. These assumptions are its **inductive bias**. A model that makes no assumptions can't generalize — it would need to memorize every possible input.

**Linear models** assume that the decision boundary (or the relationship between inputs and output) is linear in the feature space. If this assumption holds approximately, linear models are hard to beat: they're fast, interpretable, and regularization is well-understood. If the true relationship is highly non-linear and the assumption is violated, performance degrades and no amount of tuning recovers it — you need either non-linear features or a different model class.

**Tree-based models** (decision trees, random forests, XGBoost, LightGBM) make no assumptions about the distribution of features or the functional form of the relationship. They partition the feature space into rectangular regions and fit a constant within each region. They handle non-linearity and feature interactions naturally, don't require scaling, and are robust to outliers. Their weakness: they don't extrapolate — predictions outside the range of training values default to edge-of-tree behavior.

**Neural networks** are universal approximators — given sufficient capacity and data, they can approximate any continuous function. Their inductive bias is minimal in theory but in practice is shaped heavily by architecture choices (convolutions for spatial data, attention for sequences). They require large datasets, careful regularization, and extensive tuning. They're rarely the right first choice for structured tabular data.

## The Bias-Variance Tradeoff

Every model makes a tradeoff between two sources of error:

**Bias** is systematic error — the model consistently misses the true pattern because its assumptions are too strong. A linear model fit to data with a cubic relationship will always underpredict in some regions and overpredict in others, regardless of how much training data you provide. This is **underfitting**.

**Variance** is sensitivity to the specific training sample — the model fits the training data too closely, including its noise. A high-degree polynomial or an unpruned decision tree will perform perfectly on training data but poorly on new data. This is **overfitting**.

Adding complexity reduces bias (the model can represent more patterns) but increases variance (it becomes more sensitive to noise in the training data). The optimal model sits at the sweet spot — complex enough to capture the true signal, simple enough not to fit the noise.

```
Total Error = Bias² + Variance + Irreducible Noise
```

Regularization is the practical tool for managing this tradeoff: it penalizes complexity, trading a small increase in bias for a larger decrease in variance.

## Complexity vs Interpretability

In many domains, interpretability is not a preference — it's a requirement.

**Credit risk**: regulators require that adverse action notices explain why a customer was rejected. In the EU, GDPR Article 22 mandates a right to explanation for automated decisions. A black-box model that can't explain individual predictions cannot be legally deployed for credit decisions in many jurisdictions.

**Healthcare**: a model flagging patients for high-risk intervention needs to give clinicians a reason they can evaluate and override. Clinicians who don't understand a model's reasoning won't trust it.

In these contexts, interpretable models (logistic regression, scorecard, decision tree with depth ≤ 4) are often preferred not because they perform better, but because interpretability is itself a performance dimension.

## A Practical Selection Guide

Start with this sequence:

1. **Establish a naive baseline** — mean prediction, majority class, simple rule
2. **Train a regularized linear model** — logistic regression with L2 for classification, ridge for regression
3. **Evaluate the gap** — if linear model meets business requirements, stop here
4. **Add a tree-based model** — XGBoost or LightGBM with default hyperparameters
5. **Evaluate the additional lift** — is the complexity worth the gain in performance and loss in interpretability?
6. **Consider interpretability requirements** — if interpretability is mandatory and linear doesn't work, use monotone-constrained gradient boosting or a post-hoc explanation framework

Neural networks belong at step 6 or later, and only when you have enough data (typically 100k+ observations) and the feature space has structure that benefits from learned representations.

The decision to move from one level to the next should be justified by a measurable improvement on validation data — not by a preference for complexity.

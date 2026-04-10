---
slug: training-optimization
date: 2026-04-10
description: Training is minimizing a proxy for your real objective. Loss function choice, optimization landscape, and regularization as a prior belief that simpler models are more trustworthy.
authors:
  - khoi
categories:
  - Modeling
tags:
  - loss-function
  - optimization
  - regularization
---

# Training & Optimization: Minimizing a Proxy of the Real Objective

When you train a model, you're not directly optimizing what you care about. You're optimizing a loss function — a mathematical proxy for your real objective. The gap between the proxy and the objective is where models fail in production.

<!-- more -->

## The Loss Function Is More Important Than the Metric

This distinction is subtle but critical.

A **metric** measures model quality from the outside. It answers: "How well did the model do?" You compute it after predictions are made and compare against true labels. AUC, Gini, precision, recall — these are metrics.

A **loss function** guides training from the inside. It answers: "In which direction should the model update its parameters?" The gradient of the loss function tells the optimizer how to adjust weights. It must be differentiable, and it must reflect what "wrong" means in a way that steers learning in the right direction.

You optimize the loss function but report the metric. They can diverge — and when they do, the model learns to optimize something other than what you care about.

## Common Loss Functions and Their Trade-offs

**Mean Squared Error (MSE)**: penalizes large errors quadratically. A prediction error of 10 contributes 100 to the loss; an error of 1 contributes only 1. This makes MSE sensitive to outliers — a single extreme error dominates the gradient update. Use MSE when large errors are genuinely much worse than small ones.

**Mean Absolute Error (MAE)**: penalizes all errors linearly. Robust to outliers because a large error doesn't get outsized weight. The downside: the gradient is undefined at zero, which can cause instability near convergence.

**Huber loss**: combines MSE for small errors and MAE for large errors, with a threshold parameter δ that controls the boundary. Robust like MAE in the tails, smooth like MSE near zero. A good default for regression when outliers are present but not extreme.

**Binary cross-entropy**: the standard loss for binary classification. It equals negative log-likelihood under a Bernoulli distribution and is a *proper scoring rule* — the loss is minimized when the model outputs the true probability. This calibration property is why cross-entropy is preferred over alternatives like hinge loss when probability estimates matter.

**Focal loss**: a modification of cross-entropy that down-weights easy examples (those the model already classifies correctly with high confidence) and focuses learning on hard examples. Useful for severe class imbalance where the model quickly learns to predict the majority class and ignores the minority class.

## Custom Loss for Asymmetric Costs

Standard loss functions assume that all errors have equal cost. In practice, they rarely do.

In credit risk: a **false negative** (approving a loan that defaults) costs the bank the full principal. A **false positive** (rejecting an applicant who would have repaid) costs the opportunity revenue of that loan. The ratio between these costs might be 10:1 or 20:1.

A custom loss function that penalizes false negatives more heavily directly encodes this business reality:

```python
def asymmetric_loss(y_true, y_pred, fn_cost=10, fp_cost=1):
    # false negatives: predicted 0, actual 1
    fn_penalty = fn_cost * y_true * (1 - y_pred)
    # false positives: predicted 1, actual 0
    fp_penalty = fp_cost * (1 - y_true) * y_pred
    return fn_penalty + fp_penalty
```

The same effect can be achieved at the sample level by passing `sample_weight` to most sklearn-compatible models, with higher weights for the minority class or for high-cost error types.

## Optimization: Gradient Descent and Variants

Gradient descent iteratively updates model parameters by moving in the direction of the negative gradient of the loss:

```
θ ← θ − α · ∇L(θ)
```

where α is the learning rate. Choosing α matters enormously: too large and the optimizer overshoots and diverges; too small and training is impractically slow.

**Mini-batch gradient descent**: instead of computing the gradient over the entire dataset (slow) or a single example (noisy), compute it over a batch of 32–512 samples. This balances noise (which helps escape local minima) with stability.

**Adam**: maintains adaptive learning rates per parameter and uses momentum. In practice, Adam is a strong default for neural networks and gradient-boosted trees when used with libraries that expose it. It requires less learning rate tuning than vanilla SGD.

For gradient-boosted trees specifically (XGBoost, LightGBM), the optimization is different — trees are added sequentially to fit the residuals of the previous ensemble, and the learning rate controls how aggressively each tree's contribution is incorporated.

## Regularization as a Prior Belief

Regularization adds a penalty term to the loss function that discourages complexity. The interpretation: you are expressing a prior belief that **simpler models are more likely to be correct**, and the data must provide enough evidence to justify complexity.

**L2 regularization (Ridge)**: adds the sum of squared weights to the loss. All weights are penalized and shrink toward zero, but rarely reach exactly zero. This handles multicollinearity well and produces stable, small-magnitude coefficients.

**L1 regularization (Lasso)**: adds the sum of absolute weights. Some weights are driven to exactly zero, performing automatic feature selection. Useful when you suspect many features are irrelevant and you want the model to identify them.

**Elastic Net**: combines L1 and L2. Useful when you want some sparsity (L1) but also stability in the presence of correlated features (L2).

**Dropout**: during each training step, randomly set a fraction of neurons to zero. Forces the network to learn redundant representations that generalize better. Effective regularization for neural networks.

**Early stopping**: monitor validation loss during training. Stop when validation loss stops improving, even if training loss continues decreasing. The gap between the two is a direct signal of overfitting.

Regularization strength is a hyperparameter. Too little: overfitting. Too much: underfitting. The optimal value is found on the validation set, not the test set.

Training is an optimization process, but the objective is always a proxy. Understanding the gap between the loss function you optimize and the outcome you care about is what separates models that look good in notebooks from models that deliver results in production.

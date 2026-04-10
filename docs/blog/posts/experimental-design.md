---
slug: experimental-design
date: 2026-04-10
description: How you split your data determines the reliability of every conclusion you draw. Train, validation, and test sets each have a distinct role — and violating that structure silently corrupts your results.
authors:
  - khoi
categories:
  - Modeling
tags:
  - validation
  - cross-validation
---

# Experimental Design: The Split Determines the Conclusion

The most undervalued decision in a modeling project is how you divide your data. Get it wrong and every metric you report is fiction — optimistic fiction that evaporates the moment the model hits production.

<!-- more -->

## Three Sets, Three Distinct Roles

**Training set**: the data the model learns from. The model sees labels, updates parameters, and minimizes the loss function on this data.

**Validation set**: the data used to make decisions during development — which hyperparameters to use, which features to include, when to stop training. The model never trains on this data, but your choices are indirectly shaped by it.

**Test set**: the final, unbiased estimate of performance. This data should be touched exactly once — after all development decisions are finalized.

The role separation matters because each time you use the test set to make a decision, you implicitly fit that decision to the test set. Over many iterations, the test set's characteristics influence your model — and you lose your unbiased estimate.

## The No-Peeking Principle

The test set is sacred. Use it once.

This is harder to follow than it sounds. After spending weeks tuning a model, the temptation is to evaluate on the test set, notice it's slightly below target, tweak one more hyperparameter, and re-evaluate. Each re-evaluation makes the test set less independent.

**Why this matters in practice**: if a team evaluates the test set 20 times during development, they will find a configuration that appears to perform well on that specific test set — not because the model is better, but because they searched over 20 configurations and reported the best. This is selection bias operating on your holdout.

The discipline: finalize every design decision using validation data. Run the test set evaluation once, report the number, and ship the model.

## Cross-Validation

When data is scarce, a single validation split is noisy — different splits can yield meaningfully different estimates. Cross-validation addresses this by using the data more efficiently.

**k-fold CV**: split data into k folds. Train on k-1 folds, evaluate on the remaining fold. Repeat k times, average the results. Each observation appears in exactly one validation fold.

**Stratified k-fold**: maintain class proportions within each fold. Essential when the target is imbalanced — a random split might place all positive examples in one fold, making the other folds useless for calibration.

**Time-series split**: for time-ordered data, train on all data before time T, validate on data at time T+1. Advance T forward for each fold. This ensures the model never sees future data during training.

The choice of CV strategy must **simulate the deployment scenario**. If the model will be retrained quarterly and deployed for the next quarter, use a time-series split that mimics this structure. A random k-fold that shuffles time-ordered data is wrong — the model will "see the future" during training.

## Temporal Split vs Random Split

For any data with a time dimension, **random splitting is incorrect in principle**.

Imagine building a model to predict which customers will churn next month. A random split might put a customer's January record in the training set and their December record in the test set. The model can implicitly learn future information about that customer — creating leakage.

The correct approach: define a cutoff date. All data before the cutoff goes to training. All data after goes to test.

```
|---- Training ----|-- Validation --|-- Test --|
Jan                Sep              Oct         Nov
```

The gap between validation and test isn't wasted — it simulates the lag between when a model is trained and when it encounters new data in production.

## Baseline Models

Before building anything complex, establish a **naive baseline**. This is the performance of the simplest possible "model":

- For regression: predict the mean of the training target for every observation
- For classification: predict the majority class, or the class distribution (random baseline)
- For time-series: predict the previous value (random walk)
- For ranking: rank by a simple rule (e.g., total purchase value)

A baseline model serves two purposes. First, it tells you whether your model is actually learning anything — if your sophisticated model barely beats mean prediction, something is wrong. Second, it sets a concrete lower bound that your production model must exceed to justify its complexity.

If your model doesn't beat the baseline, you don't have something to deploy.

## Choosing Split Sizes

There is no universal formula, but the guiding principle is: **the validation and test sets must be large enough that your performance estimate has low variance**.

A test set of 50 observations gives you a Gini estimate with huge confidence intervals — a difference of 0.05 Gini between two models might be pure noise. A test set of 10,000 gives you a reliable estimate.

For time-series splits, the test period should match the deployment frequency. If you'll retrain monthly and the model is used for one month, your test set should be one month's worth of data.

The split ratios (80/10/10, 70/15/15) are heuristics, not rules. Let the size of your dataset and the reliability requirements of your evaluation drive the decision.

Experimental design is the foundation that all downstream conclusions rest on. If the foundation is shaky — wrong split type, contaminated test sets, mismatched CV strategy — no amount of model sophistication can compensate.

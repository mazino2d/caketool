---
slug: hyperparameter-tuning
date: 2026-04-10
description: Tuning is searching a hypothesis space — it needs a strategy. Grid search, random search, Bayesian optimization, and why over-tuning on validation data is its own form of overfitting.
authors:
  - khoi
categories:
  - Modeling
tags:
  - tuning
  - optimization
---

# Hyperparameter Tuning: Searching with a Strategy

Hyperparameters are the settings you choose before training — the number of trees, the learning rate, the maximum depth. The model doesn't learn them from data; you select them. Tuning is the process of finding the configuration that maximizes performance on held-out data.

<!-- more -->

## What Makes a Hyperparameter

A model parameter is learned from data during training — weights in a neural network, split thresholds in a decision tree. A hyperparameter controls the training process itself and must be set before any learning occurs.

Examples by model type:

| Model | Hyperparameters |
|---|---|
| Logistic Regression | Regularization strength (C), penalty type (L1/L2) |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight` |
| Random Forest | `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf` |
| Neural Network | Layer widths, learning rate, batch size, dropout rate, optimizer |

The goal of tuning is to find the hyperparameter configuration that minimizes validation error — not training error.

## Search Strategies

**Grid search**: define a discrete set of values for each hyperparameter, evaluate all combinations. Exhaustive, guaranteed to find the best combination within the defined grid, but scales exponentially with the number of parameters. With 5 hyperparameters and 5 values each, that's 5⁵ = 3,125 configurations.

**Random search**: sample configurations randomly from the defined ranges. Counterintuitively, random search often outperforms grid search in practice. The reason: in most problems, only a few hyperparameters significantly affect performance. Grid search wastes evaluations by methodically exploring unimportant dimensions. Random search covers the important dimensions more effectively with the same budget.

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {
    "n_estimators": stats.randint(100, 500),
    "max_depth": stats.randint(3, 10),
    "learning_rate": stats.loguniform(0.01, 0.3),
    "subsample": stats.uniform(0.6, 0.4),
}

search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring="roc_auc")
search.fit(X_train, y_train)
```

**Bayesian optimization** (Optuna, Hyperopt, scikit-optimize): models the relationship between hyperparameters and validation performance using a surrogate model (typically a Gaussian process or a tree-structured Parzen estimator). Uses this surrogate to decide which configuration to evaluate next — balancing exploration (trying new regions) with exploitation (refining promising regions). More sample-efficient than random search, especially when each evaluation is expensive (e.g., training a large model).

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

## Which Hyperparameters Actually Matter

Tuning all hyperparameters equally is wasteful. For XGBoost, the empirical order of importance is roughly:

1. `learning_rate` and `n_estimators` — the most impactful. These two interact: a lower learning rate generally requires more trees.
2. `max_depth` — controls tree complexity and overfitting.
3. `subsample` and `colsample_bytree` — row and column sampling, key regularization knobs.
4. `min_child_weight`, `gamma` — less impactful in most cases.

Focus your tuning budget on the top parameters. A carefully tuned `learning_rate` and `n_estimators` with default values for everything else often comes within a few percent of an exhaustively tuned model.

## The Fundamental Rule

**Tune on validation, evaluate on test. Never reverse this.**

Every time you use test set performance to make a tuning decision, you introduce leakage: the test set is no longer an independent estimate of generalization. The more decisions you make based on test set performance, the more optimistic your reported results become.

The validation set is the correct place for all tuning decisions. When using cross-validation for tuning, the inner fold is the validation set; the outer fold (or a separate holdout) is the test set.

## Over-Tuning: Overfitting to the Validation Set

This is a real phenomenon, and it's subtle. After running 500 hyperparameter trials on the same validation set, you've effectively searched over 500 configurations and reported the best. The best configuration likely exploits random variation in that specific validation set — not true generalization.

Signs of over-tuning:
- Validation performance increases smoothly with tuning iterations, but a fresh test set shows much lower performance
- The "optimal" hyperparameters are at extreme values (max or min of your search range), suggesting the range is wrong or the signal is noise

Mitigations:
- Use nested cross-validation: tune in the inner loop, evaluate in the outer loop
- Limit tuning iterations to a reasonable budget (50–200 trials)
- Reserve the test set for final evaluation only — run it once

## Early Stopping as a Special Case

For gradient-boosted trees and neural networks, `n_estimators` (or `n_epochs`) is a hyperparameter that can be tuned dynamically via early stopping: train until validation performance stops improving, then use the model at that checkpoint.

```python
model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False,
)
# model.best_iteration gives the optimal n_estimators
```

This is efficient: you don't need to search over `n_estimators` separately. Set it high, let early stopping find the right value, and tune other hyperparameters around it.

Hyperparameter tuning is search, not magic. The goal is not to find the globally optimal configuration — it's to find a configuration that generalizes reliably, found through a process that doesn't accidentally overfit to evaluation data in the process.

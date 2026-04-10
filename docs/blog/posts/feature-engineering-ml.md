---
slug: feature-engineering-ml
date: 2026-04-10
description: Features are how domain knowledge gets encoded into a model. Signal vs noise, transformation trade-offs, and the most dangerous mistake in ML — data leakage.
authors:
  - khoi
categories:
  - Modeling
tags:
  - feature-engineering
  - preprocessing
---

# Feature Engineering: Where Domain Knowledge Becomes Signal

Algorithms only see numbers. Features are the translation layer between reality and the model — how you describe each observation determines what the model can possibly learn.

<!-- more -->

## What a Feature Actually Does

A feature tells the model a story about one observation. The quality of that story determines the ceiling on model performance. You cannot learn signal that isn't in your features, no matter how powerful the algorithm.

This is why domain knowledge matters more than algorithm choice in most real-world problems. A credit analyst who knows that "number of hard inquiries in the last 6 months" is a strong default signal creates more value than choosing XGBoost over LightGBM.

**Signal vs noise**: a useful feature has high mutual information with the target. Adding more features does not automatically improve a model — irrelevant features add noise that the model must learn to ignore, and in small datasets, it often can't.

## Transformations by Data Type

### Numerical Features

**Log transform**: when a feature has a right-skewed distribution and positive values (income, loan amount, transaction value), log-transforming compresses the tail and brings the distribution closer to symmetric. Tree models don't need this (splits are invariant to monotone transforms), but linear models and neural nets benefit significantly.

**Scaling**: StandardScaler (zero mean, unit variance) vs MinMaxScaler (range [0, 1]). Tree-based models are completely invariant to scaling — skip it. Linear models and neural nets require it for stable training and meaningful regularization.

**Binning**: converting a continuous variable into discrete buckets. Useful when the relationship with the target is non-linear and you want the model to treat ranges, not values. In credit scoring, age bins (18–25, 26–35, ...) often outperform raw age because risk isn't linear across the lifespan.

### Categorical Features

| Method | When to use | Risk |
|---|---|---|
| **One-Hot Encoding** | Low cardinality (< 20 categories) | Sparse, high-dimensional for high-cardinality |
| **Target Encoding** | High cardinality | **Leakage** if not done per-fold |
| **Frequency Encoding** | When count itself is informative | Loses category identity |
| **Embedding** | Neural nets, very high cardinality | Requires sufficient data |

Target encoding replaces a category with the mean target value for that category. It's powerful but must be computed using only training data — if you compute it on the full dataset before splitting, you have data leakage.

### Temporal Features

Time-based features are where a lot of predictive power lives:

- **Lag features**: value of a variable N days/weeks ago (e.g., `balance_30d_ago`)
- **Rolling window aggregates**: mean, standard deviation, min/max over a window (e.g., `avg_spend_last_7d`, `std_spend_last_30d`)
- **Seasonality**: day of week, hour of day, month, is_holiday — behavioral patterns repeat on these cycles
- **Time since event**: days since last delinquency, days since last login

The most important constraint: **every temporal feature must be computable using only data available at prediction time**. This is point-in-time correctness.

## Leakage Prevention — The Most Important Rule

Data leakage is when information about the target variable (or future events) flows into the training features. Models trained on leaked features will appear excellent in offline evaluation and fail catastrophically in production.

**Target leakage**: a feature contains direct information about the target.

Example: predicting loan default, using `credit_limit_increase_approved` as a feature. Customers who received a credit limit increase had already passed a credit review — so the feature is downstream of the creditworthiness assessment that the model is supposed to make. It inflates AUC dramatically in training and is meaningless at prediction time.

**Point-in-time incorrectness**: a feature is computed using data that wouldn't have been available when the prediction was made.

Example: computing `total_transactions_in_2025` for a model that predicts default in March 2025. At prediction time in March, the full year's transactions don't exist yet. The feature must be `total_transactions_through_prediction_date`.

**Train-test leakage**: a preprocessing step that uses information from the entire dataset before the train/test split.

Example: fitting a StandardScaler on the full dataset, then splitting. The scaler has seen the mean and variance of the test set. The correct order is: split first, then fit the scaler on the training set only, then transform both sets.

```python
# Wrong
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # uses full dataset
X_train, X_test = train_test_split(X_scaled, ...)

# Correct
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled = scaler.transform(X_test)         # transform test
```

## Feature Selection

More features is not always better. Feature selection reduces noise, speeds up training, and often improves generalization.

**Filter methods**: compute a statistic per feature (correlation with target, mutual information, Information Value / Weight of Evidence in credit scoring). Fast, but ignores interactions between features.

**Wrapper methods**: Recursive Feature Elimination (RFE) — repeatedly train the model, drop the least important feature, repeat. Respects interactions but computationally expensive.

**Embedded methods**: L1 regularization (Lasso) drives some coefficients to exactly zero, performing feature selection during training. Tree importance scores (XGBoost `feature_importances_`) rank features by their contribution to splits.

The choice of method matters less than this: **feature selection must be done using training data only**, with the same fold structure as your cross-validation. Running RFE on the full dataset before splitting is a subtle form of leakage.

Features are the most high-leverage intervention in the modeling process. A well-constructed feature can make a simple model competitive with a complex one — and a leaked feature can make any model look better than it is.

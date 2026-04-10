---
slug: model-evaluation
date: 2026-04-10
description: Measure what you actually need to optimize. Classification metrics, calibration, slice-based evaluation, and why a beautiful aggregate metric can hide a broken model.
authors:
  - khoi
categories:
  - Modeling
tags:
  - metrics
  - evaluation
---

# Model Evaluation: Measure What You Actually Need to Optimize

A model with a high AUC score can still be a bad model. Evaluation is not about producing a number — it's about answering the question: "Will this model make the decisions we need it to make?"

<!-- more -->

## Metrics Must Align with Business Objectives

Every ML metric is a proxy for a business outcome. AUC measures the model's ability to rank positives above negatives. Precision measures the fraction of predicted positives that are actually positive. Neither of these directly measures revenue impact, default reduction, or customer satisfaction.

The connection between metric and objective must be explicit. If the business wants to catch 80% of fraudulent transactions, optimize recall — not F1, not AUC. If the business wants to minimize false alarms in a fraud review queue (because reviewers are expensive), optimize precision. If both matter equally, use F1 or set an operating point on the ROC curve that satisfies both constraints.

Define the target metric from the business requirement *before* training. If you pick the metric after seeing results, you've implicitly searched over metrics and reported the most favorable one.

## Classification Metrics

**Accuracy**: the fraction of predictions that are correct. Deeply misleading when classes are imbalanced. A dataset where 98% of transactions are legitimate produces a trivially 98%-accurate model that predicts "legitimate" for everything — and catches zero fraud. Accuracy is useful only when classes are approximately balanced.

**Precision and Recall**:

- **Precision** = TP / (TP + FP): of all predicted positives, how many are actually positive?
- **Recall** (Sensitivity) = TP / (TP + FN): of all actual positives, how many did we predict?

They trade off against each other: a higher threshold raises precision (fewer false alarms) and lowers recall (more misses). Which matters more depends entirely on the cost structure of the problem.

| Scenario | Prioritize | Reason |
|---|---|---|
| Cancer screening | Recall | Missing a positive case is far worse than a false alarm |
| Spam filtering | Precision | A false alarm (blocking legitimate email) is more costly |
| Fraud detection | Recall | Missing fraud costs more than flagging legitimate transactions |
| Credit approval | Precision | Incorrectly rejecting good customers has retention cost |

**F1 score**: the harmonic mean of precision and recall. Useful when you want a single number that balances both. Use F-beta with β > 1 when recall is more important, β < 1 when precision is.

**AUC-ROC**: the area under the Receiver Operating Characteristic curve. It measures the probability that the model ranks a randomly chosen positive example above a randomly chosen negative example. Threshold-agnostic and useful for comparing models. However, on highly imbalanced datasets, ROC curves can be optimistic — a model can achieve high AUC while performing poorly at the operating threshold.

**AUC-PR**: the area under the Precision-Recall curve. More informative than AUC-ROC when positive class frequency is very low (< 5%), because it focuses on the model's ability to find positives without being inflated by the large pool of true negatives.

**KS Statistic and Gini**: standard in credit risk. KS is the maximum separation between the cumulative distribution of scores for positives and negatives. Gini = 2 × AUC − 1. A Gini of 0 means the model has no discrimination power; 1 means perfect discrimination. Industry benchmarks: Gini > 0.30 is useful, > 0.50 is strong.

## Regression Metrics

**MAE (Mean Absolute Error)**: average absolute deviation. Interpretable in the same units as the target. Robust to outliers.

**RMSE (Root Mean Squared Error)**: square-root of average squared error. Penalizes large errors more than MAE. Sensitive to outliers — a few large misses can dominate the metric.

**MAPE (Mean Absolute Percentage Error)**: expresses error as a percentage of the actual value. Intuitive for business stakeholders ("we're off by 8% on average") but undefined when actual values are zero and biased toward under-prediction.

**R²**: the proportion of variance in the target explained by the model. Ranges from 0 to 1 for well-behaved models, but can be negative for models worse than mean prediction. Easy to misinterpret — a high R² does not mean the model is useful; it means the model explains variance, which is valuable only if variance is what you care about predicting.

## Calibration

A model is well-calibrated if its predicted probabilities match empirical frequencies. If a model assigns a probability of 0.3 to 1,000 events, approximately 300 of them should actually occur.

Poor calibration means the model's probability outputs can't be trusted directly. A model that systematically underestimates risk (outputs 0.1 for events that occur 30% of the time) will cause decision systems that rely on those probabilities to systematically approve too many high-risk customers.

Check calibration with a **reliability diagram** (calibration curve): bin predictions by probability, plot average predicted probability vs observed frequency. A well-calibrated model lies on the diagonal.

Tree-based models and SVMs are often poorly calibrated and benefit from post-hoc calibration using Platt scaling (logistic regression on the outputs) or isotonic regression.

## Slice-Based Evaluation

Aggregate metrics hide distributional failures. A model with Gini 0.42 overall might have Gini 0.18 for customers aged 18–25 — a segment that is systematically underserved or misclassified.

Always break down performance by:

- **Demographic segments**: age group, geography, income band
- **Temporal slices**: performance by month to detect drift over time
- **Product segments**: different loan products, different customer tiers
- **Score bands**: performance at the top, middle, and bottom deciles

Slice analysis finds two types of problems: **model bias** (the model performs worse for a subgroup, often correlated with protected characteristics) and **distribution shift** (the model's performance has degraded in a specific segment that the training data didn't represent well).

A model that performs well on the aggregate but poorly on a critical slice is not a good model — it's a model that will produce unacceptable outcomes for a subset of real users.

## The Metric Hacking Trap

Optimizing a metric hard enough always finds a way to game it. A model tuned to maximize AUC on a fixed test set over many iterations will find configurations that exploit quirks of that specific sample — and generalize poorly.

The defense: keep the test set truly held out, report only what the model achieves on first contact with it, and evaluate the model's business outcomes (actual default rates, actual revenue impact) after deployment — not just its offline metrics.

The goal of evaluation is not a number on a leaderboard. It's confidence that the model will make the right decisions when it encounters new data in production.

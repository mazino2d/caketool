---
slug: productionization-monitoring
date: 2026-04-10
description: Every model decays. Training-serving skew, data drift vs concept drift, PSI, and why deployment is the beginning of a model's life — not the end.
authors:
  - khoi
categories:
  - Modeling
tags:
  - mlops
  - monitoring
  - drift
---

# Productionization & Monitoring: Models Decay Over Time

Deploying a model is not the finish line. It's the starting gun for a continuous process of monitoring, maintenance, and eventual replacement. Every model degrades — the question is how fast, and whether you'll know before it does damage.

<!-- more -->

## Training-Serving Skew

The most immediate production failure is **training-serving skew**: the features computed at serving time differ from the features computed during training.

This happens more often than you'd expect. A feature pipeline that runs in notebooks during development gets re-implemented in a production system — and the implementations diverge. Edge cases are handled differently. Aggregation windows are off by one day. A join uses a different key.

The result: the model at serving time receives inputs from a different distribution than what it was trained on. Even if the model is theoretically correct, it's receiving the wrong data.

The prevention: **the training pipeline and the serving pipeline must share code, not just logic**. If you compute `avg_spend_last_30d` during training, the exact same function — same code, same time boundaries, same handling of nulls — must compute it at serving time. Drift from two separate implementations is an engineering problem, not a modeling one.

Validate by logging a sample of serving features and comparing their distribution to the training feature distribution regularly.

## Data Drift vs Concept Drift

Once in production, models face two distinct types of decay:

**Data drift** (covariate shift): the distribution of inputs P(X) changes, but the relationship between inputs and outputs P(Y|X) remains the same.

Example: a credit model trained on data from 2022 encounters 2024 customers who have, on average, lower incomes and higher debt loads due to macroeconomic changes. The model's features look different than what it was trained on. Even if the relationship "lower income + higher debt → higher default probability" still holds, the model may be operating in regions of the feature space it rarely saw during training.

Data drift can be detected **without labels** — you only need to compare input distributions between training and serving, which is available in real time.

**Concept drift**: the relationship P(Y|X) changes. The same input now produces a different outcome.

Example: a fraud detection model trained before a new fraud vector emerges. Fraudsters adapt. The patterns that previously indicated fraud (unusual location, unusual amount) no longer reliably identify the new fraud type. The model's learned mapping from features to fraud probability is no longer correct.

Concept drift requires **ground truth labels** to detect — you need to observe actual outcomes, which introduces a lag. A loan default model trained in January won't accumulate enough 90-day default labels to measure performance degradation until April.

| Type | What changes | Detectable without labels? | Lag to detect |
|---|---|---|---|
| Data drift | P(X) | Yes | Hours to days |
| Concept drift | P(Y\|X) | No | Weeks to months |

## Monitoring Metrics

**Population Stability Index (PSI)**: the standard metric for measuring distributional shift in credit risk. Computes the difference between two distributions using a symmetric KL-divergence-like formula:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Interpretation:
- PSI < 0.1: no significant change, monitor normally
- 0.1 ≤ PSI < 0.2: moderate change, investigate
- PSI ≥ 0.2: significant shift, model review required

Monitor PSI at two levels:
1. **Prediction distribution**: the distribution of model scores over time. A shift here is an immediate signal that something has changed.
2. **Feature distribution**: PSI per input feature. This helps pinpoint *which* feature is driving the shift, enabling targeted investigation.

**Performance metrics over time**: Gini decay and KS decay are the primary indicators for credit models. Plot Gini by cohort (the month the loan was originated) over time. A steady decline indicates model aging; a sudden drop may indicate a population shift or a data pipeline issue.

**Outcome rate monitoring**: track the actual positive rate (default rate, fraud rate) over time against the model's expected positive rate. Divergence indicates concept drift.

## Retraining Strategy

There is no universal retraining schedule. The right strategy depends on how fast the problem environment changes:

**Schedule-based**: retrain monthly, quarterly, annually. Simple to operationalize. Risk: the schedule may be too slow (model degrades between cycles) or too fast (frequent retraining adds cost without benefit).

**Performance-based**: retrain when Gini drops below a threshold (e.g., drops 5 points from baseline). Requires real-time label collection, which may not be feasible for long-horizon outcomes like 90-day default.

**Drift-based**: retrain when PSI exceeds a threshold. Proactive — you don't wait for performance to degrade. The risk: data drift doesn't always cause performance degradation. Retraining too aggressively wastes resources.

In practice, most teams use a combination: scheduled retraining as a baseline, with drift-based triggers for emergency retraining when PSI alarms fire.

## MLOps Foundations

Sustainable production ML requires three capabilities:

**Model versioning**: every deployed model has a version identifier, and every prediction can be traced back to the model version that made it. Libraries like MLflow make this straightforward. Without versioning, debugging production failures is guesswork.

**Feature lineage**: every feature in the serving pipeline has a documented source — which table, which transformation, which time window. When a feature distribution shifts, lineage tells you where to look.

**Reproducibility**: given the same training data and the same code, you can reproduce the same model. This requires pinning library versions, fixing random seeds, and version-controlling preprocessing code alongside model code.

These are not ML innovations — they're software engineering practices applied to the ML context. The reason they're often missing from ML systems is that models are developed in notebooks (not in version-controlled pipelines) and deployed through ad-hoc processes.

## The Production Mindset

Deployment is a transition, not an endpoint. After a model goes live:

- It will encounter distributions it hasn't seen
- The world it was trained on will change
- The business requirements it serves will evolve

A model without monitoring is a ticking clock. A model with monitoring is a system you can maintain. Treat production deployment as the beginning of the model's lifecycle — budget time and engineering capacity for monitoring, retraining, and replacement as part of the original project scope.

The projects that skip this step are the ones that get an urgent message six months later: "The model is doing something weird."

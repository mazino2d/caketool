---
slug: problem-formulation
date: 2026-04-10
description: A wrong problem definition makes a correct model useless. How to translate a business problem into a well-formed ML problem — target variable, unit of observation, horizon, and falsifiability.
authors:
  - khoi
categories:
  - Modeling
tags:
  - problem-formulation
  - fundamentals
---

# Problem Formulation: Getting the Question Right Before Choosing a Model

Getting the problem wrong is worse than getting the model wrong. A misdefined problem produces a model that performs well on the wrong objective — and that failure is invisible until it hits production.

<!-- more -->

## The Most Expensive Mistake in ML

A team spends three months building a churn prediction model. It achieves AUC 0.81. It gets deployed. Six months later, the retention team reports that the model's predictions have zero correlation with actual revenue impact.

The problem: the target variable was "cancelled subscription" — a label that fires when a user manually clicks "cancel." But the business defines churn as "stopped using the product." Many users stop using the product months before they cancel. The model was predicting an administrative event, not the behavior the business cared about.

The model was correct. The problem was wrong.

## Translating Business Problem → ML Problem

The translation has four components:

**Target variable: what are we predicting?**

Be precise. "Predict default" is not a target variable. "Binary indicator: did the customer miss a payment of more than 30 days within the 90-day window after origination?" is a target variable. The exact definition determines labeling logic, class balance, and what "good performance" means.

**Unit of observation: what does one row represent?**

One customer? One transaction? One customer-month? One loan application? The unit of observation determines how the training dataset is constructed, and misdefining it causes data leakage or incorrect aggregations. In credit risk, the common unit is *one loan at origination* — not one customer (a customer may have multiple loans).

**Horizon and latency: how far ahead, and how fast?**

"Predict default in the next 90 days" is different from "predict default in the next 12 months." A longer horizon means weaker signal but more time to intervene. Latency matters too: does the decision need to be real-time (in-session loan approval) or can it run overnight (monthly portfolio review)?

**Success metric from the business side, not the ML side:**

The CFO wants to reduce the default rate by 2 percentage points while keeping the approval rate above 60%. That's the business KPI. Gini coefficient and AUC are proxies for that KPI. Don't optimize the proxy and forget the KPI — they can diverge.

## Problem Taxonomy

Choosing the right ML problem type is part of formulation:

| Type | When to use | Example |
|---|---|---|
| **Regression** | Output is continuous | Predict loan amount a customer will repay |
| **Classification** | Output is a discrete category | Approve / reject / refer to analyst |
| **Ranking** | Order matters, not absolute score | Rank customers by collection priority |
| **Anomaly Detection** | Normal behavior is well-defined, anomalies are rare | Flag fraudulent transactions |

The same business problem can be framed as multiple types. Fraud detection is often framed as classification (fraud / not fraud), but a ranking formulation (rank transactions by fraud probability, review top N%) is often more operationally useful when the review team has a fixed capacity.

## The Falsifiability Requirement

A well-formed ML problem must be **falsifiable**: there must be a clear, observable way to determine whether the model was right or wrong.

"Predict customer satisfaction" is not falsifiable unless satisfaction is measured (through a survey, an NPS score, a return rate). Without a measurable ground truth, there is no training signal and no way to evaluate performance.

If you cannot describe how you would know the model was wrong, you don't have an ML problem yet.

## Anti-Patterns to Avoid

**Jumping to model selection before defining the unit of observation.** This is the most common mistake. Once you choose a model architecture, you implicitly constrain the problem structure. Define the data schema first.

**Using available labels instead of the right labels.** Labels that are easy to collect (button clicks, system events) are not always the right target. The right target is whatever the business decision is actually trying to optimize.

**Ignoring the deployment context.** A model that requires 48-hour latency cannot serve a real-time credit decision. The deployment constraint belongs in the problem definition, not in a post-hoc engineering discussion.

**Conflating prediction horizon with label collection delay.** If you predict 90-day default, you need 90 days of history after origination to collect labels. This has direct implications for how fresh your training data can be, and how long before you can evaluate a deployed model.

Problem formulation is not a one-page document you write and forget. It is a living agreement between the data team, the business stakeholders, and the engineers who will deploy the model. Every ambiguity in the formulation becomes a bug somewhere in the pipeline.

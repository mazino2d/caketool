---
slug: modeling-philosophy
date: 2026-04-10
description: Models don't need to be right — they need to be useful. Why parsimony matters, and the three questions you must answer before building anything.
authors:
  - khoi
categories:
  - Modeling
tags:
  - philosophy
  - fundamentals
---

# The Philosophy of Modeling: A Controlled Approximation

> *"All models are wrong, but some are useful."* — George Box

The first question beginners ask is: "Which model is best?" That's the wrong question. The right question is: "Best at what, and for which problem?"

<!-- more -->

## What Is a Model?

A model is a **controlled approximation** of reality. Not a replica. Not the truth. A simplified representation of the world, precise enough to support a specific decision.

A map is a model of geography. It omits elevation changes, road textures, building colors — yet it's enough to navigate. The map is "wrong" in a literal sense, but useful in a practical one.

ML models work the same way. A logistic regression predicting loan default doesn't capture the full complexity of human financial behavior. But if it helps a bank reduce its default rate by 2% over the current heuristic, it's useful — and that's what matters.

## "Correct" vs "Useful"

This is where many data scientists get lost: they pursue a "correct" model instead of one that's "good enough for the specific decision at hand."

A practical example: in a credit scoring problem with moderate data (100k records, 50 features), a **carefully tuned logistic regression** often matches or outperforms XGBoost in production — and it's far easier to explain to a compliance team. A neural network with a complex architecture might achieve AUC 0.003 higher on the test set, but it can't be deployed because no one can explain why customer A was rejected while customer B was approved.

The "most correct" model by metric is not always the best model for the problem.

## The Parsimony Principle

**Occam's Razor** applied to modeling: between two models that explain the data equally well, prefer the simpler one.

Not because simplicity is a goal in itself. But because:

- **Fewer assumptions** → fewer ways to be wrong in unexpected ways
- **Better generalization** → less overfitting to training data
- **Easier to maintain** → when business requirements change, a simple model is easier to update
- **Easier to debug** → when the model fails in production, root cause analysis is faster

Parsimony doesn't mean always using a linear model. It means: **don't add complexity unless there is clear evidence that the complexity is necessary.**

Complexity must earn its way in.

## Before You Start: Three Questions

Before opening a notebook, answer these three questions:

**1. What exactly are we approximating?**

Describe the input-output relationship precisely. "Predict the probability that a customer will miss a payment within the next 90 days, based on transaction history and demographic data." Not just "build a credit model."

**2. What does "good enough" mean?**

Set a concrete, measurable success criterion *before* training anything. "A Gini coefficient of at least 0.35 on the holdout set, with precision at top 10% above 40%." Not "the model should be accurate."

**3. Who will use this output, and to make which decision?**

Will a credit analyst review each prediction manually? Or will an automated system approve or reject in real time? The answer directly shapes the threshold, latency requirement, and interpretability standard.

These aren't formalities. They define the scope, success criteria, and constraints of the entire pipeline — from data collection to deployment.

## Three Practical Consequences

This philosophy has three concrete implications:

**Baseline first, complexity later.** Always start with the simplest possible model. If a linear model achieves Gini 0.30 and the business needs 0.35, there's a clear reason to add complexity. If it already hits 0.36, stop.

**Metrics must come from the problem.** AUC is not the objective — it's a proxy. The objective is a business outcome. Don't chase a pretty number at the expense of the decision the model is actually serving.

**The best model is the one that gets used.** A 0.95 accuracy model that no one trusts enough to deploy is worthless. An 0.82 accuracy model that the team understands, trusts, and can maintain has real value.

This is the mental framework for the rest of this series. Every subsequent topic — problem formulation, feature engineering, evaluation, monitoring — comes back to the same question: "Which approximation is good enough for this decision?"

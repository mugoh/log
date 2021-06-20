---
layout: post
comments: true
title: "Maximum entropy Inverse Reinforcement Learning"
date: 2020-09-15 12:00:00
tags: paper-summary inverse-rl imitation-learning
---

> A way to resolve ambiguity in choosing decision distributions during imitation learning by use of maximum entropy
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


### Introduction
- Frames Imitation Learning as having the goal to predict the behaviour and decisions an agent would choose e.g., the motions a person would take to grasp an object or theroute a driver would take to get from home to work.
  - This requires reasoning about action consequences far into the future

- The unknown reward the agent will be optimizing for is assumed to be linear in the features.
- In matching the reward value of demonstrated behaviour, **maximum entropy** is applied to resolve ambiguity in choosing a distribution over decisions.
- Create a probabilistic model that normalizes globally over behaviours - an extention to chain conditional random fields that incorporates dynamics of the planning system and ex-tends to the infinite horizon
- Author's efforts motivated by real-world rouring preferences of drivers. They are able to infer future routes and destinations based on partial trajectories.
- Max-entropy approach also deals with noise and imperfect behaviour in demonstraions, which would introduce uncertainty.


### Conclusion
- Presents an iRL approach that solves for ambiguities in previous approaces.

- Future work: Add contextual factors e.g Time of day, and regional factors e.g Avoid some road in rush hour, in feature space

- Application: Driver prediction and route recommendation


### 2. Background
#### Ambiguity in Recovered Policies

Referenced previour work:
1. Maximum Margin Prediction
- Learn a reward function by minimizing loss expressed as disagreement between learned, and agent policy.

2. Feature Expectations - Matching the observed policy features to learner's behaviour
- Both Feature matching and maximum margin prediction will have many policies leading to same feature counts. With sub-optimal behaviour in demontrations, many policy mixutes will satisty feature matching.

3. Max-Entropy iRL
- Max entropy iRL, resolving ambiguity, results in a single stochastic policy.
It resolves the  ambiguity by choosing the distribution that doesnot exhibit any additional preferences beyond matching feature expectation

- The partition function (with reward parameters) will not converge for infinite horizon problems with **zero-reward absorbing states**.


#### Non-deterministic Path distributions**


- General MDP Actions produce non-deterministic transitions between states according to state transition distributions *T*.

- Max-ent used is conditioned on $T$ and constrained to match feature expectations.

They use an indicator function $Iζ∈ o$ which is $1$ when $ζ$ is compatible with $o$ and $0$ otherwise. This distribution over paths is approximated with the assumption that  transition randomness has a limited effecton behavior and that the partition function is constant for all $o ∈ T$. (See equation in p3)
For stochastic policies, the partition function converges.



#### Learning from demonstrated behaviour.
Aims to maximize likelihood of observed data under that max-entropy distribution.

$$
θ∗= \argmax_{\theta} L(θ) = \argmax_θ \sum_{examples}\logP( ̃ζ|θ, T)
$$


The gradient is the difference between expected empirical feature counts and the learner’s expected feature counts, which(learner's) can be expressed in terms of expected statevisitation frequencies. Feature expectations match at maxima.

The algorithm **relies on emueration of all paths** to compute the expected state frequencies. To avoid this the work uses a forward-backward algorithm to compute the partition function of each state and action.







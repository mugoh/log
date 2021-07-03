---
layout: post
comments: true
title: "Safe Model-based Reinforcement Learning with Stability Guarantees"
date: 2020-10-18 12:00:00
tags: paper-summary safety
---

> Defines safety as stability guarantees. Use Lyapunov stability verification to show how to use dynamic models to obtain policies with stability guarantees.
> Also show one can safely collect data
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}



- Year: 2017
- [https://arxiv.org/pdf/1705.08551](https://arxiv.org/pdf/1705.08551)
- Understanding: "***"

This work defines safety as stability guarantees. Use Lyapunov stability verification to show how to use dynamic models to obtain policies with stability guarantees.

Also show one can safely collect data

### 1. Introduction

- Authors describe a *region of attraction* (ROA), a subset of the state-space, so that any trajectory starting within it stays within it until convergence.
- Safety here is described as the ability to recover from exploratory actions to a safe state
- By starting with a safe initial policy and collecting more data in the safe region, authors expand the estimate of the safe region
- Contribution: Safety + stability in learning

Mentioned previous safety work, authors say:

- Requires an accurate probabilistic model of the system
- Is task specific & requires system reset
- Model-based RL safety focus is on state constraints or MPC with constraints
- GP-modeled-system approaches estimate region of attraction but don't update the policy

- Authors guarantee *safety guarantees in terms of stability*. Improve $\pi_\theta$ and increase the safe region of attraction without leaving it:

> Specifically, starting from a policy that is known to stabilize the system locally, we gather data at informative, safe points and improve the policy safely based on the improved model of the system and prove that any exploration algorithm that gathers data at these points reaches a natural notion of full exploration.

**Conclusion**: Constraint safety in terms of stability. Provide *theoretical* safety and exploration guarantees

### 2. Preliminaries

- The authors use a deterministic, discrete time system

$$x_{t+1} = f (x_t , u_t ) = h(x_t , u_t ) + g(x_t , u_t )$$

The model consists of a prior model $h(x_t , u_t )$ and a priori of unknown model errors  $g(x_t , u_t )$

- The safety constraint is defined on the state-divergence occuring when leaving the region of attraction(ROA).
- So exploratory actions shouldn't leave the region of attraction and adapting the policy shouldn't deacrease the region of attraction(ROA).
- **Assumption**: The models $h$ and $g$ are **Lipschitz continuous**
- Lyapunov function $v : X → \mathbb{R}_{≥0}$ is used to define the region of attraction. It's step deacrease for states guarantees eventual convergence to origin (a minimum)

### 3. Theory

- Here, the assumptions are applied to RL. This involves:
    - Computing the RoA for fixed $\pi$
    - Optimizing $\pi$ to expand ROA
    - Prove how to safely learn the dynamics
    - Introduce their algorithm "Safe Lyapunov learning"

### 4. Experiments

- The algorithm sacrifices exploration for safety
- Algorithm implementation on github [https://github.com/befelix/safe_learning](https://github.com/befelix/safe_learning)
- For policy updates, the goal is to adapt the policy for minimum cost while ensuring the safety constraint is not violated.
- GP confidence intervals are used to verify safety
- Their way of computing the ROA suffers from the curse of dimensionality, but they say it's not necessary to do real-time policy udpates. i.e., Any of the provenly safe policies can be used for an arbitrary no. of time steps
- Not scaled to high-dimension systems -- this would consider adaptive discretization for the verification

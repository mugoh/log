---
layout: post
comments: true
title: "Safe Exploration in Continuous Action Spaces"
date: 2020-09-27 12:00:00
tags: paper-summary safety
---

> Solving for complex RL tasks without a reward function while providing human feedback (oversight) on less than 1% of agent interaction with environment.
> Addresses deploying RL on a system where critical constraints shouldn't be violated e.g datacenter-cooling. Adds a safety layer to solve an action formulation for each state.

<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


- [https://arxiv.org/pdf/1801.08757.pdf](https://arxiv.org/pdf/1801.08757.pdf)
- Year: 2018

### 1. Introduction
This work defines a goal for maintaining zero constraint violations throughout the learning process. It's simpler for discrete than continous action spaces.

- An observable quantity is to be kept constrained e.g temperature and pressure thresholds in cooling, angles in robot joints

- Also removes the necessity of using trajectories generated using a known behaviour policy that can be mathematically  described - focuses on physical systems (actions have short-term effects)

- The safety layer is directly on top of the policy that corrects actions when needed. It finds minimal change in actions that meet safety constraints

**Conclusion**: This work provides state-based action correction to accomplish zero-constraint violations in constrained areas. It promotes efficient exploration by action guiding towards safe policies. Offers advantage in that one does'nt need to know the behaviour policy used in trajectory generation.

- The work uses a constrained markov decision process (CMDP). It has a set of immediate consrained functions (similar to Constrained Policy Optimization)


**Terms**

- The contrain cost is defined:
$$
C = {c_i :S × A → R | i ∈ [K]}
$$
based on which they define a set of safety signals ${c̄ i : S → R | i ∈ [K]}$

- Assume a deterministic transition $s' = f(s, a)$ (for systems they use in this work)


### 2. State-wise Constrained Policy Optimization


### 3. Linear safety signal model
- During the initial stages, the agent must violate constraints enough for exploration.

- To handle this, the authors incorporate some prior knowledge using single-step dynamics and learn a model of the *immediate* (not full transition model) constraint functions $c_i(s, a)$.

    $$
	\bar{c}_i (s') \overset{ \Delta }{=} c_i(s, a)  \approx \bar{c}_i(s)  + g(s; w_i)^T a
    $$

(See eq at 5 p3)

A 1st order apprx of $c_i(s, a)$ w.r.t $a$


> To comment in AISE linear models question:
Potentially helpful: "Linear approximations of non-linear physical systems prove accurate and are well accepted in many fields. For the comprehensive study, see [link to Enqvist, Martin. Linear models of nonlinear systems]"


- $g(s, w_i)$ is pre-trained by minimizing (see eq 2 pg 3) $\arg\min_{w_i}\sum_{(s, a,s')}\bar{c}_i(s') -  (\bar{c}_i(s)  + g(s; w_i)^T a)^2$  using $D = {(s_j , a_j , s'_j )}$ randomly collected until a time-limit or a constraint violation.

- Authors mention there's no advantage of continual training of $g(s, w_i)$


### 4. The Safety Layer via Analytical Optimization

- The safety layer is placed on top of the policy net, solving an action correction optimization on each forward pass.
It alters the action as little as possible in the Euclidean norm, to meet the constraint.

$$
	a* = argmin_a 1/2 || a − μ_θ (s)|| 2
	s.t \text{the linear model above meeting the constaint}
$$


**Optimizing for this**
- Assumpation: Only 1 constraint is active at a time.

- For multiple constraints though, a joint model is learnable e.g Distance between to walls(2 const) = min distance between them (1 constraint).

- The authors use a closed form solution where there's an optimal Langrange multiplier $\lambda_i^*$ associated with the $i_{th}$ constraint. It's a function of $g(s; w_i)$

- The optimal action is finally expressed:
$$
	a* = μ_θ (s) − λ^∗_i * g(s; w i^∗ ),
	where i ∗ = \arg\max_i λ^∗_i
$$




- In implementation, authors say **it's 3 lines of code**, computationally efficient and differentiable almost everywhere. 

- When $c_i (s, a)$ is used as an NN ${λ i > 0}$ are now hyper-parameters. However, this has performance drawbacks from the linear version.




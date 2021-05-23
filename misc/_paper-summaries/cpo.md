---
layout: post
comments: true
title: "Constrained Policy Optimization"
date: 2020-10-20 12:00:00
tags: paper-summary safety reward-design
---

> Maximizing rewards while satisfying safety constraints using a constrained MDP
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


- Year: 2017
- [https://arxiv.org/pdf/1705.10528](https://arxiv.org/abs/1705.10528)

Proves a bound relating the expected returns of two policies to an average divergence between them.

### 1. Introduction

In critical domains, a natural way to incorporate safety in exploration is through constraints. This presents the Constrained Markov Decision Process (CMDP) where agents must satisfy on auxilliary costs.

The authors bound the difference between the reward or cost of 2 different policies, and the result, of independent interest, tightens known bounds for policy search using trust regions. Authors say this will increase both the reward and satisfaction of other safety costs.

**Conclusion**

CPO can train neural net policies on high-dimensional constrained control tasks by maximizing reward while satisfying safety constraints.

### 2. Some Related Work

- Objective trading return with risk while learning these trade-off coefficients.
- Using intrinsic fear, instead of constraints, to motivate the agent to avoid rare but catastrophic events
Authors say their work is first to guarantee constraint satisfaction throughout training and with arbitrary policy classes (including neural nets)
- Authors also mention the discounted future state distribution, which is interesting to note.

$$
d^π (s) = (1−γ) \sum_{ t=0}^{\inf} γ^t P (s_t = s|π)
$$

They use this to express the difference in performance between two policies.

### 3. Constrained Markov Decision Process (CMDP)

A CMDP has constraints that restrict the set of allowable policies for that MDP. It has a set of costs  $C_i : S × A × S → R$ . Each of these constraints $C_i$ has a limit $d_i$.
The expected discounted $\pi$ return with respect to the cost is expressed similar to the reward function:

$$\pi_c =	\sum γ^t C _i (s_t , a_t , s_{t+1})
$$

Then on-policy advantage, Q and Value functions $$V^{\pi}_{C_i}, Q^{\pi}_{C_i}$$, and $$A^{\pi}_{C_i}$$ are defined for the auxilliary costs.

### 4. Constrained Policy Optimization

In doing gradient updates we restrict the divergence between the new and old polices to some limit $D(π, π_k ) ≤ δ.$

For a CMDP, instead of optimizing over $Π_θ$, we optimize over $Π_θ ∩ Π_C$ :

$$	J_{C_i} (π) ≤ d_i where:     i = 1, ..., m
$$

$$D(π, π_k ) ≤ δ$$

To approximate the above equation, the authors replace the objective and constraints with surrogate functions. The surrogates are estimated using samples from $Π_k$.

- From the choice of the surrogate, it's possible to bound the update's worst-case performance and worst-case constraint violation with values that depend on a hyperparameter of the algorithm.

### 5. Policy Performance Bound

- Authors say there's a relation between the constrained returns between two arbitrary polices and the average divergence between them.
- Use Total Variational Divergence $D_{TV}$ between the policies as a bound, which is the total variational divergence between the action distributions at state $s$.

    $$D_{TV}(\pi'|\pi)[s] = \frac{1}{2} \sum | π' (a|s) − π(a|s) |$$

- The $D_{TV}$ is related to the KL divergence $D_{KL}$ by:

$$D_{TV} = \sqrt{0.5 D_{KL}}$$

- Since CPO is based on TRPO, it inherits its monotonic policy improvement and performance guarantee.

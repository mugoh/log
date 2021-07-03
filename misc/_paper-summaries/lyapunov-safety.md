---
layout: post
comments: true
title: "A Lyapunov-based Approach to Safe Reinforcement Learning"
date: 2020-10-09 12:00:00
tags: paper-summary safety constrained-mdp
---

> This safety approach is under a Constrained Markov Decision Process (CMDP). It's a method for constructing Lyapunov functions which provided guarantee of global safety of a behaviour policy during training via a set of local linear constraints

<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


- Year: 2018
- [https://arxiv.org/pdf/1805.07708](https://arxiv.org/pdf/1805.07708)

This safety approach is under a Constrained Markov Decision Process (CMDP). It's a method for constructing Lyapunov functions which provided guarantee of global safety of a behaviour policy during training via a set of local linear constraints

### 1. Introduction

- Lagrangian method, as way of solving CMDPs runs into the saddle point problem when learning the Lagrange multiplier and doensn't guarantee behaviour policy safety during each training iter.
- An improvement to this, using a stepwise surrogate constraint only dependant on current (s, a) pair is conservative and has suboptimal performance.
- Authors say analysing convergence of constrained policy optimization (CPO) is challenging and applying it to other RL algorithms (besides TRPO) non-trivial.
- Instead of a handcrafted Lyapunov candidate, this work uses an LP-based algorithm to construct Lyapunov functions w.r.t. generic CMDP constraints.
- Algorithms presented:
    - Safe policy Iteration (SPI)
    - Safe Value Iteration (SVI)
    - Safe DQN & safe DPI (for large s/a spaces - scalable)

**Conclusion**:

- Derives an LP-based method to generate Lyapunov functions guaranteeing feasibility and optimality in the algorithm.
- Lyapunov approach guarantees safety and robustness learning in RL
- Future work: Apply this to Policy Gradient (PG) methods and compare to CPO on continous action spaces. (ii) Test in the real world

### 2. Preliminaries

- A constrained MDP has these additional factors on the unconstrained MDP: $d(x) ∈ [0, D_{max }]$ - immediate constraint cost; and
$d_0 ∈ R ≥0$ -  upper-bound on the expected cumulative (through
time) constraint cost

- The CMDP $OPT$ problem is formulated:

Given an initial state $x_0$ and a threshold $d_0$ , solve

$$\min π∈∆ C_π (x_0 ) : D_π (x_0 ) ≤ d_0 .$$

If there is a non-empty solution, the optimal policy is denoted by $π^∗$ .

$$D_π (x_0 ) ≤ d_0 \text { is a safety constraint, with }D_π \text {given as}: 
\\
D_π (x_0 ) := E [\sum_{t_0}^{T* - 1} d(x_t) | x_0, \pi ]
\\
\text {where  T* is the First step of a terminal state by } \pi
$$

**This paper has plenty of functions new to me.** Won't list them all. Refer to paper for better proof of concepts.

### 3. A Lyapunov Approach to Solve CMDPs

The main goal is to construct a Lyapunov function $L ∈ L_{π_B} (x_0 , d_0 )$ such that $L(x_0 ) ≤ d_0$

- The constraint value function $D_{\pi^*}$ w.r.t. optimal policy $π^∗$ can be transformed into a Lyapunov function that is induced by some baseline  $π_B$ , i.e., $L_{\epsilon^∗} (x) ∈ L_{π_B} (x_0 , d_0 )$
- A cost shaping term $\epsilon(x_t)$ is used and a **contraction mapping** $T_{\pi_B, d + \epsilon}$
- In conclusion, the Lyapunov function $L_{\epsilon^∗}$ is a uniform upper-bound to the constraint cost, i.e.,

$$L_{\epsilon^∗} (x) ≥ D_{π_ B} (x)$$

- The authors enforce a condition to make the optimal policy $\pi ^*$ to be close to  $\pi_B$ such that the set of $L_{\epsilon^∗}$ − [induced policies] contains an optimal policy
- The solution to $OPT$ is $V^*(x)$, where $V^*(x)$ is a value function expressed

$$T V = V (x), ∀x ∈ X \ is \ unique$$

- If $\pi_B$ satisfies the distance to $\pi^*$ constraint, at  $x = x_0, V*(x)$ gives the OPT solution

### 4. Safe RL using Lyapunov functions

- $\epsilon^\star$ is approximated with an auxilliary constraint cost, with the largest cost giving a better chance of including  $\pi^*$ in the set of feasible policies.
- In safe policy iteration (SPI), updating the Lyapunov function involves bootstrapping. $L_{\sim{\epsilon}}$ is updated at each iteration using a recomputation involving the total visiting probability from the initial state to any state. This visiting probability function is a maximizer of the objective.
- SPI and SVI update the Lyapunov function based on the best baseline policy.
- They have monototic $\pi$ improvement and have convergence

### 5. Lyapunov-based Safe RL Algorithms

- DQN and DPI replace the policy and value updates with function approximators
- Updates utilize **policy distillation** and the **Jensen-Shannon divergence** between policy updates

Both SDQN and SDPI sample mini-batches from buffer to minimizes MSE loss in value function updates.

- Safe Q-learning SDQN: It uses a constraint value network $Q_D$ and a stopping time value network $Q_T$ to make a Lyapunov function estimate. The auxilliary constraint cost $\epsilon^\prime$ is function approximated.
- SDPI: Value estimation is done using policy evaluation

### 6. Experiments

- The algorithms are tested in a 2D grid planning problem - the goal is to reach a given box, with constraint being to avoid hitting obstacles.
- Number of obstacles is constrolled by a density  $\rho \in (0, 1).$
- There's explicit knowledge of the transition probability and reward function.
- Lyapunov approaches perform safe RL without a known model of the env, with deep function approximators.
- SPI has good performance
- SVI and Lagrangian approaches degrade as $\rho$  grows
- SDQN and SQPI guarantee that, on finding a safe policy, all other updates remain safe. But Lagrangian approaches achieve worse rewards, violate constraints and don't guarantee safety when the constraint threshold $d_0$ is big(5)

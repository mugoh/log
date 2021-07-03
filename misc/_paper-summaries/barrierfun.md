---
layout: post
comments: true
title: "Barrier Functions for Safe-critical Continuous Control Tasks "
date: 2020-10-11 12:00:00
tags: paper-summary safety 
---

> A Gaussian Process(GP) is used to model the system uncertainties and dynamics.
> Guarantees safety regardless of RL algorithm used and shows exploration efficiency
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


- Year: 2019
- [arxiv.org/pdf/1903.08792](http://arxiv.org/pdf/1903.08792)

Proposes a controller architecture combining:

- Model-free RL-based controller with
- a model-based controller using control barrier functions (CBF)
- Online learning of unknown system dynamics

A Gaussian Process(GP) is used to model the system uncertainties and dynamics.

Guarantees safety regardless of RL algorithm used and shows exploration efficiency

### 1. Introduction

- Authors argue that model-free safety approaches to RL (like reward-shaping, policy optimization with constraints) don't guarantee safety during learning. It needs env interactions, meaning violation during intital learning stages. I think that's obvious.
- Model based approaches have used Lyapunov or Model Predictive Control (MPC) for learning of safe system dynamics but don't adrress perfomance optimization (Which, from [https://arxiv.org/pdf/1805.07708](https://arxiv.org/pdf/1805.07708), don't think is completely true) and exploration.
- Existing model-free work that incorporates model info for safe exploration use back-up safety controllers, limiting learning/exploration efficiency
- This work RL-CBF provides for integration of model-free RL algorithms with CBFs for safety and exploration efficiency.
- CBFs require a nominal model but ensure safety of non-linear systems and exploration of the policy space.

**Conclusion**: RL-CBF guarantees safery and improves exploration. Integrates with new RL algorithms. It allows online learning improvements

### 2. Preliminaries

- Authors model the time evolution of the system using the equation:

$$s_{t+1} = f (s_t ) + g(s_t )a_t + d(s_t )$$

- $f$ and $g$ compose a known nominal model of the dynamics, and $d$ represents the unknown model.
- A Gaussian Process (GP) model is used to estimate the unknown system dynamics  $d(s)$ from the data.
- The GP is batch trained with latest ~1000 data points as an estimating to bypass a matrix inversion when estimating the GP uncertainty.

### 3. Control Barrier Functions

- The algorithm attempts to ensure exploration occurs within some safe set.
- CBF uses a Lyapunov-like argument to ensure forward invariance of the safe set under controlled dynamics
- There are derivations on how CBF (or sometimes, almost) encodes safety, guiding exploration of RL with CBF and ensuring computational efficiency with MLP.

- RL-CBF Algorithm

![End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks]({{'assets/images/paper_summaries/barrier-fun_alg.png' | relative_url}})
{: style="width: 80%;" class="center"}
*Figure 1: The RL Continuous Barrier Functions algorithm*

### 4. Experiments

- Authors use a 5 car problem, and control the 4th car to maintain a 2 metre distance from the rest.
- RL-CBF manages to avoid collisions in this, while standard TRPO & DDPG do not.
Experiment 2 is a pendulum.
- RL CBF controllers never leave the safe region

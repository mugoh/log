---
layout: post
comments: true
title: "Model-Based Value Expansion for Efficient Model-Free RL"
date: 2020-12-08 12:00:00
tags: paper-summary model-based-rl
---

> Controls uncertainty in the model by only allowing imagination to a fixed depth.
> The learned dynamics are used in a model free RL algorithm to improve value estimation, in turn reducing sample complexity.
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}




- Year: 2018
- [https://arxiv.org/pdf/1803.00101](https://arxiv.org/pdf/1803.00101)

Model free RL proposes incorporating imagined model data with a notion of uncertainty to accelerate learning of continuous control tasks. But relies on heuristics that limit the usage of the dynamics model.

This work presents *model-based value expansion (MVE)*, which controls for uncertainty in the model by only allowing imagination to a fixed depth

The learned dynamics are used in a model free RL algorithm to improve value estimation, in turn reducing sample complexity.

**Notable**: Learning with stochastic Value function Ensemble, (Background 2.2) has good background on MVE. (After reading it, I thought I missed out some details on its working)

Related Paper: [Sample-Efficient Reinforcement Learning
with Stochastic Ensemble Value Expansion](https://arxiv.org/pdf/1807.01675.pdf)

## Introduction

- Complex dynamics require high capacity models, which are prone to overfitting.
- Expressive value estimation MF tasks achieve good performance but have poor sample complexity, while MB methods show efficient learning but struggle on complex tasks.
- This work combines MB and MF techniques to support complex non-linear dynamics while reducing sample complexity
- MVE uses a dynamics model to simulate short-term horizon and Q-learning to estimate the long-term value beyond the simulated horizon.
- So it splits value estimates into a near MB component(which requires no differentiable dynamics) and a distant MF component

> imagination-augmented agents (I2A) offloads all uncertainty estimation and model use into an implicit neural network training process, inheriting the inefficiency of model-free methods

- Incorporating the model into Q-value target estimation only requires the model to make forward predictions.
- Unlike stochastic value gradients (SVG), this makes no differentiability assumptions about underlying dynamics, which may include non-differentiable phenomena like [contact interactions](https://abaqus-docs.mit.edu/2017/English/SIMACAEITNRefMap/simaitn-c-contactoverview.htm#:~:text=Contact%20interactions%20in%20a%20model,normal%20direction%20to%20resist%20penetration.)
- Authors say this work (from the experimental results) can outperform fully MF algorithms and prior MB-MF accelerating approaches

**Discussion**:

- This work introduces MVE, an algorithm for incorporating predictive models of system dynamics into model-free value function estimation
- Existing approaches use stale imagination data in the buffer or imagine past accurate horizons
- MVE offers model trust upto a horizon H and utilizes the model upto that extent
- It's state dynamics prediction also enables on-policy imagination via the TD-k trick, starting with off-policy data.

## 2. Model-Based value exapansion (MVE)

- MVE improves value estimates for a policy by assuming use of an approximate dynamical model $f: \mathcal{S} \times A \rightarrow \mathcal{S}$ and a true reward function $r$
- The model is assumed accurate upto a depth $H$
- Using the imagined reward $\hat{r}_t = r(\hat{s_t}, \pi({\hat{s}_t)})$ obtained using the imagined state $\hat{s}_t$, the MVE estimate for the value function of a given state $V^\pi (s_0)$ is defined:

$$\hat{V}_H(s_0) = \sum_{t=0}^{H-1} \gamma^t \hat{r}_t + \gamma^H\hat{V}(\hat{s}_H)$$

The state value estimate at $s_0$ is composed of the learned dynamics prediction $\hat{r}_t$ + the tail estimated by $\hat{V}$

- Since $\hat{r}_t$ is derived from actions $\hat{a}_t =$ $\pi(\hat{s}_t)$, this is an on-policy use of the model and MVE doesn't need importance weights.
- Assuming the model is almost ideal the MVE relates to the value function  MSE in this way: (This consideres the H depth model accuracy assumption), where imagined states $\hat{s_t}$, actions and rewards equal the actual states $s_t$, actions rewards while $t< H$

    $$\hat{V}_H(s_0) - V^\pi(s_0) \approx \gamma ^H\left(\hat{V}_H(s_H) - V^\pi(s_H)\right)$$

- The authors use a TD-k error to alleviate the distribution mismatch between the imagined state distribution and the true state distribution (see subsection 3.1 for complete details)

### 2.1 Deep RL implementation

The implementation uses an actor $\pi_\theta$ and a critic $Q\varphi$, but the actor may be replaced for $\pi(s) = \arg \max_a Q_\varphi(s, a)$

- Rollouts are imagined with the target actor
- The authors do not have an imagination buffer for simulated states, which are instead generated on the fly *(*from sampling any point upto H imagined steps into the future)
- A real transition $\tau_0 = (s_{t-1}, a_{t-1}, r_{t-1}, s_0)$is first sampled. The model $\hat{f}$ is used to generate $\hat{s_t}$ and $\hat{r_t}$. Since $\pi_{\hat{\theta}}$ changes during the joing optimization of $\theta$ and $\varphi$, the simulated states are discarded imediately after the batch. A stochastic grad step  $\nabla_\varphi$ is then taken to minize the Bellman error of $Q_\varphi$

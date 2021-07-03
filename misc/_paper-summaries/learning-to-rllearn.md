---
layout: post
comments: true
title: "Learning to Reinforcement Learn"
date: 2020-11-11 12:00:00
tags: paper-summary meta-rl meta-learning
---

> RL requires a massive amount of training data. Thus the objective is to develop methods that adapt quickly to new tasks.
> This work extends the approach of RNN supporting meta-learning in a fully supervised context. It gives a system trained on one RL algorithm, but whose recurrent dynamics implement a second, separate procedure.
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}

- Year: 2017
- [https://arxiv.org/pdf/1611.05763](https://arxiv.org/pdf/1611.05763)

RL requires a massive amount of training data. Thus the objective is to develop methods that adapt quickly to new tasks.

This work extends the approach of RNN supporting meta-learning in a fully supervised context. It gives a system trained on one RL algorithm, but whose recurrent dynamics implement a second, separate procedure.

## 1. Introduction

The key concept is to use standard deep RL techniques to train an RNN in such that the RNN implements its own, free-standing RL procedure. This can enable the secondary RL procedure have task adaptiveness and sample efficiency

**Conclusion**:
Deep meta-RL involves:

1. Use of deep RL to train an RNN
2. Training set as a series of interrelated tasks
3. Net input includes action selected and reward from previous time interval.

This allows the RNN to learn to implement a second RL procedure. This learned algorithm builds in **domain-appropriate biases** allowing greater efficiency than the general purpose one.

- A system trained using model-free RL can develop behaviour emulating model-based control

## 2. Background

- At an architectural level, meta-learning is conceptualized to involve 2 systems: a lower-level one responsible for adapting to new tasks and a slower higher-level one that works across tasks to tune and improve the lower system
- In a regression meta-learning task, the RNN receives an input $x$ at each step and an input $y$ from the previous step. Learning of new taks, here, is within the dynamics of the RNN (not backgropagation) and will continue with the weights frozen

### 2.1 Deep metal-RL

- The agent received inputs indicating the action from the previous time-step and the associated reward.
- As with the supervised case, the dynamics of the RNN implement a different RL algorithm from that used to train the weights. Authors mention:
    - Policy update procedure (inluding features like the effective learning rate)
    - Implementing its own exploration approach
    - Learning can occur after the weights are held constant

### 2.2 Algorithm

- $\mathcal{D}$  being a prior over MDPs, meta-RL should learn a prior-dependant RL algorithm and averagely perform well on MDPs drawn from  $\mathcal{D}$ or with slight modifications
- At each episode, a new MDP task $m \sim \mathcal{D}$ and its initial state are sampled, and the agent's internal state reset
- At each step, an action $a$ is executed as a function of the whole history  $\mathcal{H}_t =  \{x_0, a_0, r_0, \dots, x{t-1}, x_t, a_t, r_t\}$  of the agent interacting with MDP $m$ during the current episode since the RNN was reset
- After training, the policy is frozen and evaluated on MDPs with same (or almost) distributions as $\mathcal{D}$
- Activations will be changing due to the env inputs and hidden RNN state
- The internal state is reset at the beginning of each evaluation episode
- Since the policy is history dependant, it is able to adapt and deploy a strategy optimizing for rewards in a new MDP (task)

## 3. Experiments

- **Objective**: Identify if meta-RL:
    - could be to learn an adaptive balance between exploration and exploitation
    - give efficient algorithms that capitalize on task structure
- *The RNN feeds into a softmax (Discrete actions)*
- All inputs include a scalar for the previous episode reward and a one-hot encoded representation for the sampled action
- The architectures used are described in experiments

- Experiments are done on bandits with dependant arms & with independent arms and learning abstract task structure
- In the abstract task structure, the agent learns to bind a selected image with a rewarding role, after observing the outcome of the first episode trial
- In one-shot navigation the goal is reached in ~100 steps in the first time in an episode and ~30 steps in later visits. Authors say meta-RL allows the agent to infer the optimal value function following initial exploration, with one LSTM (use stacked LSTMs) providing info about the current relevant goal location to the policy-outputing LSTM

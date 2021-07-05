---
layout: post
comments: true
title: "RL2: Fast Reinforcement Learning via Slow Reinforcement Learning"
date: 2020-11-11 12:00:00
tags: paper-summary meta-rl
---

> It represents RL as an RNN and learns from data. The RNN recevies transitions and retains the state across episodes.

<!--more-->

{:class="table-of-content"}
* TOC
{:toc}



- Year: 2016
- [https://arxiv.org/pdf/1611.02779](https://arxiv.org/pdf/1611.02779)

RL requires a huge number of trials. Animals can learn from few trials using prior knowledge. This paper seeks to bridge this gap.

It represents RL as an RNN and learns from data. The RNN recevies transitions and retains the state across episodes.

The encoded weights of the RNN are learned through a "slow" RL algorithm. Activations of the RNN store the state of the "fast" RL algorithm

## 1. Introduction

- Deep RL has been able to accomplish many tasks, but at the expense of high sample complexity. e.g To master a game, RL would need thousands of hours, in contrast to 2 hours from a human player. Authors equate this to lack of a good prior.
- Bayesian RL enables incorporation of prior knowledge, but computational update is tractable in only simple cases. Algorithms then incorporate Bayesian and domain-specific ideas to lower computational and sample complexity but they make environmental assumptions.
- This paper takes a different approach to hand-designing a domain-specific RL algorithm:
    - They average the objective across all possible MDPs, reflecting the prior to be distilled into an agent
    - The agent is an RNN receiving past $(r, a, d)$ plus observations
    - Internal state is preserved across episodes, giving capacity to perform learning in its own hidded activations
- This work is tested on multi-armed bandits and vision-based navigation

**Discussion**: Instead of designing an RL algorithm ourselves, this paper suggests learning the algorithm end-to-end. The "fast" algorithm has its state stored in the RNN activations, while the RNN weights are learned by a slow algorithm.

Suggested improvement: Better policy architecture for long-horizon settings in the outer loop. Architectures that exploit the problem structure.

## 2. Formulation

- Authors define a *trial* which consists of 2 episodes ($n=2$)
- For each trial, for each episode, a separate $s_0$ is drawn from the inital state distribution
- Input to the policy is a concatenation of  $(s', a, r, d)$
- The policy, conditioned on hidden state $h_{t+1}$ generates the next hidden state $h_{t+2}$ and $a_{t+1}$. Policy hidden state is preserved between episodes, not between trials
- Objective maximizes the total discounted reward under a full trial
- **MDPs change across trials** — the agent acts according to the belief over the current MDP
- Hence the agent integrates all info and continually adapts its strategy

## 3. Policy

### 3.1 Representation

- The policy is represented by an RNN (Authors use **GRU**)
- Each timestep it receives an input $(s, a, r, d)$ embedded by a function $\phi(s, a, r, d)$
- The GRU output is fed to an FC followed by a softmax which forms the distribution over actions
- Authors use TRPO and GAE

## 4. Evaluations/Experiments

- Evaluation done on:
    - Multi-armed bandits (MAB) and tabular MDPs.
    - Visual navigation task (For high dimensional tasks)

### MAB

- Investigates if policy learns tradeoff between exploration and exploitation
- RL$^2$ achieves almost-as-good performance to other MAB strategies -- UCBI, Thomposon sampling  $\epsilon$-greedy, Gittins Index, Greedy
- RL$^2$ **outperforms them in finite horizon settings** because they miminize asymptotic regret rather than finite horizon regret

### Tabular MDPs

- For fewer episodes($<50$) RL$^2$ outperforms the listed strategies. But this diminishes as the number of episodes $n$ increase
- Authors argue the advantage for a small $n$ is from the need of aggressive exploitation.
- As $n$ increases, the RL problem in the outer loop becomes difficult to solve

### Visual Navigation

- Introduces high dimensional spaces
- There's reduction in trajectory length between the first 2 episodes suggesting the agent learns to use info from previous episodes
- But in larget mazes ratio of improved trajectories is lower: authors say the agent has not learnt to act optimally
- However, the agent doesn't always reuse prior info and sometimes "forgets" where the target was, and continues to explore
- The RL$^2$ formulation constructs a POMDP in the outer loop, the underlying MDP unobserved by the agent. But authors define the inner problem (where agent sees $n$ trials) as an MDP

(see [learning to RL learn]({{ '/paper-summaries/learning-to-rllearn' | relative_url }}) for another visual navigation task)

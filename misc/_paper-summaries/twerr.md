---
layout: post
comments: true
title: "Trial without Error: Towards Safe Reinforcement Learning via Human Intervention"
date: 2020-09-28 12:00:00
tags: paper-summary safety human-in-the-loop
---

> This work trains a learner to imitate a human intervention's decisions.
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}


- [https://arxiv.org/pdf/1707.05173.pdf](https://arxiv.org/pdf/1707.05173.pdf)
- Year: 2017


### 1. Introduction

The goal of HIRL is to enable the agent to learn without a single catastrophe.

- This is achived in simple envs but with more complex catastrophes, the number is significantly reduced, though not to zero.
- Authors compare this to giving an agent large negative rewards for catastrophic events without stopping them (Reward Shaping) - the agent still causes them later, unlike HIRL.
- A catastrophic action is one the human overseer deems unacceptable.

### 2. Specification of HIRL

- At each time-step, the human observes the current state s and proposed action a. If (s, a) is catastrophic, the human sends a safe action $a^*$ to the env instead.
- The new reward $r=R(s,a^\ast)$ is replaced with a penalty $r^*$.
- During the oversight period, $(s, a)$  are stored with a binary label whether a human blocked it or not.
- This dataset trains a classifier by Supervised Learning (SL) ("Blocker") to imitate the human's blocking decisions.
- The oversight lasts until the Blocker performs *well* (no def given) on a held-out dataset, after which the blocker takes over for the rest of the time.
- Oversight is done in multiple phases because of the $(s, a)$ distirbution shift during learning - which also happens on transfer to another agent.
- This works with any RL algorithm.

### 3. Discussion

- Oversight was done for 4.5 hours on most Atari but this was insufficient for more complex tasks (Road Runner).
- HIRL can't scale to complex tasks as the human time-labelling cost would be infeasible.
- Authors conclude by giving points on possible ways to address this of which some are:
    - Data efficiency in data size needed to train the Blocker, and also for agent to learn unsafe actions.
    - Seeking catastrophies during oversight. i.e., Maximixize human labelling on (s, a) that are actually unsafe
    - Active Learning - The agent requests feedback on unsure states, instead of labelling over an entire duration.

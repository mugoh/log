---
layout: post
comments: true
title: "Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning"
date: 2020-10-01 12:00:00
tags: paper-summary safety
---

> Avoids manual resets and unsafe actions by learning a policy that can predict when the agent is about to enter a non-reversible state
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}



- [https://arxiv.org/pdf/1711.06782](https://arxiv.org/pdf/1711.06782)
- Year 2017

RL requires repeated attempts and env resets but not all tasks are easily reversible without human intevention.

This work learns a forward and a reset policy. The reset policy resets the env for a subsequent attempt and determines when the forward policy is about to enter a non-reversible state.

Reduces: No. of manual resets, unsafe actions resulting in non-reversible states

### 1. Introduction

The bottleneck for many real-world tasks is that the data collection stops when the agents awaits a manual reset.

- We often write large negative rewards to avoid such dangerous states which doesn't scale to complex envs
- The goal of this work is to learn not only how to do a task but also to undo it. They use the intuition that reversible actions are safe, and can be undone to get to the original state
- The reset policy restricts the agent to states it can return from.
- It resets the env between episodes and ensures safety from unrecoverable states.
- Adding uncertainty into the value functions of both policies makes the process risk-aware, balancing safety and exploration.

**Conclusion**: This work provides for auto resets between trials and early aborts to avoid unrecoverable states.

- Reduces the number of manual resets - a number of them still needed to learn some tasks
- X Treats cost of all manual resets as equal. e.g Breaking glass cost equals displacing a block
- All experiments simulated. For real-world the challenge will be auto identifying when agent resets.

### 2. Comparison to Curicullum Generation

- Learning a reset policy is related to culicullum generation: the reset controller is engaged in increasingly distant states, providing a curriculum for the reset policy. Authors explore how prior methods did curriculum generation using a separate goal setting policy.
- In contrast, this work allows the reset policy to terminate an episode but doesn't set an explicit goal
- Authors say the training dynamics of this reset policy is similar to reversed curicullum learning.

3. Continual Learning with Joint Forward-Reset Policies

- The forward policy reward $r_f (s, a)$ is the env task reward.
- Authors use off-policy actor-critic algorithms.
- The reset policy reward $r_r (s)$ is designed large for states with high density under initial state distribution.

    **Early Aborts**

    - Transitioning from the Forward's (policy) final state to an initial state is challenging in difficult envs.
    - Irrecovarable states have low reset policy (Reset) value function.
    - So if the Reset's  $Q$  value for a proposed action is too small, an early abort is done.
    - The $Q$ function is the probability that a reset will succeed
    - Early aborts prevent the Forward from entering unsafe states and can be an alternative to manual constraints in robotic tasks

    **Hard Resets** (env.reset)

    - Early aborts are aimed at minimizing this.
    - Authors identify an irreversible state as one where the Reset fails to reset after N (0 - 8) continous episode attempts. Increasing N decreases no. of hard resets substantially.
    - Authors define safe states as  $S_{reset} ⊆ S$ and say the current $s$ is irreversible if the set of states visited by the reset policy over the past  $N$ episodes is disjoint from $S_{reset}$. (no element in common)

    **Value Function Ensembles**

    - Authors use an ensemble of $Q$ functions for better uncertainty estimates of the true value function on unseen states.
    - With this ensemble distribution 3 abort strategies are proposed:
        1. Optimistic - Early abort if all Q values < $Q_{min}$
        2. Realistic - Mean Q < $Q_{min}$
        3. Pessimistic - Any Q < $Q_{min}$

### 4. Experiments

- Lower value of abort threshold $Q_{min}$ reduces no. of hard resets
- This work reduces manual resets while achieving comparably ~equal returns with the standard RL with only a Forward policy
- Increasing N (episodic reset attempts) reduces hard resets
- Q Ensembles are key for this to work
- Induces a curicullum allowing solving of (otherwise unsolvable: *Authors only test on peg insertion*) difficult tasks.

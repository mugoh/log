---
layout: post
comments: true
title: "Variational Discriminator Bottleneck: Constraining Information Flow"
date: 2020-09-15 12:00:00
tags: paper-summary inverse-rl
---

> Contraining information flow from the inputs to the discriminator to encourage representation learning
> overlapping the data and the generator's distribution
<!--more-->

{:class="table-of-content"}
* TOC
{:toc}

- [https://arxiv.org/abs/1810.00821](https://arxiv.org/abs/1810.00821)


### Introduction
- A discriminator that achieves very high accuracy can produce relatively uninformative gradients, but a weak discriminator can also hamper the generator’s ability to learn.

- Paper proposes a regularization technique for adversarial learning, which constrains information flow from the inputs to the discriminator using a variational approximation to the information bottleneck.

- Enforcing a constraint on the mutual info between the input observations and the discriminator's(disc) internal representation encourages the disc to learn a representation that overlaps the data and the generator's (Gen) distr. This modulates the disc accuracy & maintains informative Gen grads.

#### 2. Related Work
- Other regularization methods for GAN stability & convergence - Explicit gradient penalty, architectural constraints, but this (Information bottleneck to discriminator) ignores irrevalant cues

- The performance of policies trained through adversarial methods still
falls short of those produced by manually designed reward functions

- Adding a variational bottleneck produces comparable results with those of manually engineered reward functions, authors say. It allows the generator to focus on improving the most discerning differences between real and fake samples.

- Compressed input representation can improve generalization by ignoring
irrelevant distractors present in the original input.



###  Incorporating the Bottleneck

- Introduce an encoder $E(x|z)$.
- The encoder maps features $X$ to a latent distr over $Z$.
- Enforce an upper bound $I_c$ on the mutual information between the encoding and the original features $I(X, Z)$


### Experiments
- Gradient Penalty GP applied to the discriminator helps prevent exploding gradients.
- Variational Discriminator Bottleneck (VDB) prevents vanishing gradients. So the two methods can be complementary.

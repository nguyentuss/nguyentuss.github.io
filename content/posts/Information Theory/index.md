---
title: Information Theory
description: null
date: 2025-04-20 11:34:08+0700
image: cover.jpg
categories: null
tags:
- Probability
- Math
math: true
---

## KL Divergence

**Kullback-Leibler (KL) divergence** measures how one probability distribution diverges from a second, expected distribution.

### Definition

For discrete distributions $P$ and $Q$ over a set $\mathcal{X}$:

$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

For continuous distributions with densities $p(x)$ and $q(x)$:

$$
D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} \, dx
$$

### Intuition

* KL divergence quantifies the **extra information** needed when using $Q$ instead of the true distribution $P$.
* It is **asymmetric**, so generally:

$$
D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)
$$

### Properties

* **Non-negativity**:
  $$
D_{KL}(P \parallel Q) \ge 0
$$
  Equality holds if $P = Q$.

* **Not a true distance metric** (not symmetric and no triangle inequality).

### Example

Let:

$$
P(x) = 
\begin{cases}
0.4 & x_1 \\
0.3 & x_2 \\
0.3 & x_3
\end{cases}
\quad
Q(x) = 
\begin{cases}
0.5 & x_1 \\
0.2 & x_2 \\
0.3 & x_3
\end{cases}
$$

Then:

$$
D_{KL}(P \parallel Q) = 
0.4 \log(0.4/0.5) +
0.3 \log(0.3/0.2) +
0.3 \log(0.3/0.3)
$$

## Entropy

**Entropy** is a measure of the **uncertainty** or **information content** of a random variable.

### Definition

For a discrete random variable $X$ with distribution $P(x)$:

$$
H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)
$$

If $\log$ is base 2 → Entropy is in **bits**.  
If $\log$ is base $e$ → Entropy is in **nats**.

### Intuition

* Higher entropy = more **surprise** or **uncertainty**.
* Lower entropy = more **predictable** outcomes.

### Properties

* Non-negativity:
  $$
H(X) \ge 0
$$

* Maximum entropy when all outcomes are equally likely:
  
  $$
H_{\text{max}} = \log n
$$
  
  where $n$ is the number of outcomes.

* Additivity for independent variables:
  
  $$
H(X, Y) = H(X) + H(Y)
$$

### Example

#### Fair coin:

$$
P(H) = 0.5,\quad P(T) = 0.5
$$

$$
H(X) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = 1 \text{ bit}
$$

#### Biased coin:

$$
P(H) = 0.9,\quad P(T) = 0.1
$$

$$
H(X) = -0.9 \log_2 0.9 - 0.1 \log_2 0.1 \approx 0.47 \text{ bits}
$$

### Applications

* **Information Theory**: Data compression, transmission.
* **Machine Learning**: Decision trees, classification loss functions.
* **Physics**: Thermodynamic disorder.
* **Cryptography**: Measuring randomness and uncertainty.

### Connection to KL Divergence

KL divergence can also be expressed in terms of entropy:

$$
D_{KL}(P \parallel Q) = -H(P) - \sum_x P(x) \log Q(x)
$$

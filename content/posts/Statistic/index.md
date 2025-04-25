---
title: Statistic
description: null
date: 2025-04-20 12:03:54+0700
image: cover.jpg
categories:
- Math
tags:
- Math
- Probability
math: true
---

## Introduction

In the section [Univariate Models](https://nguyentuss.github.io/p/univariate-models/) and  [Multivariate Models](https://nguyentuss.github.io/p/multivariate-models/), we assumed all the parameters $\theta$ is known. In this section, we discuss how to learn these parameters from data.
The process of estimating $\theta$ from $\mathcal{D}$ is call **model fitting**, or **training**, and is at the heart of machine learning. There are many methods for producing such estimates, but most boil down to an optimization problem of the form.
$$
\widehat{\theta} = \arg\min_{\theta} \mathcal{L}(\theta)
$$
where $\mathcal{L(\theta)}$ is some kind of loss function or objective function. We discuss several different loss functions in this chapter. In some cases we also discuss how to solve the optimization problem in closed form. In general, however we will need to use some kind of generic optimization algorithm, which we will discuss in [Optimization](https://nguyentuss.github.io/p/optimization/).
In addition to computing a **point estimate $\widehat{\theta}$**. We discuss how to model our uncertainty or confidence in this estimate. In statistics, the process of quantifying uncertainty about an unknown quantity estimated from a finite sample of data is called **inference**.

---

## Maximum likelihood estimation (MLE)

The most common approach to parameter estimation is to pick the parameters that assign the highest probability to the training data; this is called **maximum likelihood estimation** or **MLE**. We give more details below, and then give a series of worked examples.

### Definition

We define the MLE as follows:
$$
\widehat{\theta}_{\text{mle}} \triangleq \arg\max_{\theta} p(\mathcal{D} \mid \theta)
$$

We usually assume the training examples are independently sampled from the same distribution, so the (conditional) likelihood becomes

$$
p(\mathcal{D} \mid \theta) = \prod_{n=1}^{N} p(y_n \mid x_n, \theta)
$$

This is known as the i.i.d assumption, which stands for “independent and identically distributed”. We usually work with the log likelihood, which is given by
$$
\ell(\theta) \triangleq \log p(\mathcal{D} \mid \theta) = \sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)
$$

This decomposes into a sum of terms, one per example. Thus, the MLE is given by
$$
\widehat{\theta}_{\text{mle}} = \arg\max_{\theta} \sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)
$$

Since most optimization algorithms (such as those discussed in [Optimization](https://nguyentuss.github.io/p/optimization/)) are designed to *minimize* cost functions, we can redefine the **objective function** to be the (conditional) **negative log likelihood** or **NLL**:

$$
\text{NLL}(\theta) \triangleq -\log p(\mathcal{D} \mid \theta) = -\sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)
$$
Minimizing this will give the MLE. If the model is unconditional (unsupervised), the MLE becomes

$$
\widehat{\theta}_{\text{mle}} = \arg\min_{\theta} -\sum_{n=1}^{N} \log p(y_n)
$$
since we have outputs $y_n$ but no inputs $x_n$. In statistics, it is standard to use $y$ to represent variables whose generative distribution we choose to model, and use $x$ to represent exogenous inputs (coming from outside the system), which are given but not generated.
Alternatively we may want to maximize the **joint** likelihood of inputs and outputs. The MLE in this case becomes

$$
\widehat{\theta}_{\text{mle}} = \arg\min_{\theta} -\sum_{n=1}^{N} \log p(y_n, x_n \mid \theta)
$$

### Justification for MLE

There are several ways to justify the method of MLE. One way is to view it as simple point approximation to the Bayesian posterior $p(\theta|\mathcal{D})$ using a uniform prior (A **uniform prior** is a type of **prior distribution** used in **Bayesian statistics**, where we assume that **all values within a certain range are equally likely** before we observe any data).
In particular, suppose we approximate the posterior by a delta function, $p(\theta \mid \mathcal{D}) = \delta(\theta - \widehat\theta_{\text{map}})$, where $\widehat{\theta}_{\text{map}}$ is the posterior mode, given by

$$
\widehat{\theta}_{\text{map}} = \arg\max_{\theta} \log p(\theta \mid \mathcal{D}) = \arg\max_{\theta} \log p(\mathcal{D} \mid \theta) + \log p(\theta)
$$

If we use a uniform prior, $p(\theta) \propto 1$, the MAP estimate becomes equal to the MLE, $\widehat\theta_{\text{map}} = \widehat\theta_{\text{mle}}$.
Another way to justify the use of the MLE is that the resulting predictive distribution $p(y \mid \widehat\theta_{\text{mle}})$ is as close as possible (in a sense to be defined below) to the **empirical distribution** of the data. In the unconditional case, the empirical distribution is defined by

$$
p_{\mathcal{D}}(y) \triangleq \frac{1}{N} \sum_{n=1}^{N} \delta(y - y_n)
$$

We see that the empirical distribution is a series of delta functions or “spikes” at the observed training points. We want to create a model whose distribution $q(y) = p(y \mid \theta)$ is similar to $p_{\mathcal{D}}(y)$.
A standard way to measure the (dis)similarity between probability distributions $p$ and $q$ is the **Kullback-Leibler divergence**, or **KL divergence**. We give the details in [here](https://nguyentuss.github.io/p/information-theory/#kl-divergence), but in brief this is defined as

$$
D_{\text{KL}}(p \parallel q) = \sum_{y} p(y) \log \frac{p(y)}{q(y)}
$$
$$
= \sum_{y} p(y) \log p(y) - \sum_{y} p(y) \log q(y)
$$
$$
= \underbrace{-\mathbb{H}(p)}_{\text{entropy}} + \underbrace{\mathbb{H}_{\text{ce}}(p,q)}_{\text{cross-entropy}}
$$

where $\mathbb{H}(p)$ is the entropy of $p$ ([Cross Entropy](https://nguyentuss.github.io/p/information-theory/#entropy)), and $\mathbb H_{\text{ce}}(p,q)$ is the cross-entropy of $p$ and $q$. One can show that $D_{\text{KL}}(p \parallel q) \geq 0$, with equality if $p = q$.
If we define $q(y) = p(y \mid \theta)$, and set $p(y) = p_{\mathcal{D}}(y)$, then the KL divergence becomes

$$
D_{\text{KL}}(p \parallel q) = \sum_{y} \left[ p_{\mathcal{D}}(y) \log p_{\mathcal{D}}(y) - p_{\mathcal{D}}(y) \log q(y) \right]
$$
$$
= -\mathbb{H}(p_{\mathcal{D}}) - \frac{1}{N} \sum_{n=1}^{N} \log p(y_n \mid \theta)
$$
$$
= \text{const} + \text{NLL}(\theta)
$$

The first term is a constant which we can ignore, leaving just the NLL. Thus minimizing the KL is equivalent to minimizing the NLL which is equivalent to computing the MLE.
We can generalize the above results to the supervised (conditional) setting by using the following empirical distribution:

$$
p_{\mathcal{D}}(x, y) = p_{\mathcal{D}}(y \mid x)p_{\mathcal{D}}(x) = \frac{1}{N} \sum_{n=1}^{N} \delta(x - x_n)\delta(y - y_n)
$$

The expected KL then becomes

$$
\mathbb{E}_{p_{\mathcal{D}}(x)} \left[ D_{\text{KL}}\left( p_{\mathcal{D}}(Y \mid x) \parallel q(Y \mid x) \right) \right] = \sum_{x} p_{\mathcal{D}}(x) \left[ \sum_{y} p_{\mathcal{D}}(y \mid x) \log \frac{p_{\mathcal{D}}(y \mid x)}{q(y \mid x)} \right]
$$
$$
= \text{const} - \sum_{x, y} p_{\mathcal{D}}(x, y) \log q(y \mid x)
$$
$$
= \text{const} - \frac{1}{N} \sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)
$$

Minimizing this is equivalent to minimizing the conditional NLL in Equation.

### Example: MLE for the Bernoulli distribution

Suppose $Y$ is a random variable representing a coin toss where the event $Y=1$ corresponds to heads and $Y=0$ corresponds to tails. Let $\theta=p(Y=1)$ be the probability of heads. The probability distribution for this r.v is the Bernoulli, which we introduced in [Univariate Models](https://nguyentuss.github.io/p/univariate-models/).
The NLL for the Bernoulli distribution is given by

$$
\text{NLL}(\theta)=-log \prod_{n=1}^{N} p(y_n|\theta)
$$
$$
= -log \prod_{n=1}^N \theta^{\mathbb{I}(y_n=1)}(1-\theta)^{\mathbb{I}(y_n=1)}
$$
$$
=-\sum_{n=1}^{N}[\mathbb{I}(y_n=1)\log(\theta)+\mathbb{I}(y_n=0)\log(1-\theta)]
$$
$$
= -[N_1\log(\theta)+N_0\log(1-\theta)]
$$

where we have defined $N_1 = \sum_{n=1}^{N} \mathbb{I}(y_n = 1)$ and $N_0 = \sum_{n=1}^{N} \mathbb{I}(y_n = 0)$, representing the number of heads and tails. (The NLL for the binomial is the same as for the Bernoulli, modulo an irrelevant $\binom{N}{c}$ term, which is a constant independent of $\theta$.) These two numbers are called the **sufficient statistics** of the data, since they summarize everything we need to know about $\mathcal{D}$. The total count, $N = N_0 + N_1$, is called the **sample size**.
The MLE can be found by solving $\frac{d}{d\theta} \text{NLL}(\theta) = 0$. The derivative of the NLL is

$$
\frac{d}{d\theta} \text{NLL}(\theta) = \frac{-N_1}{\theta} + \frac{N_0}{1 - \theta}
$$

and hence the MLE is given by

$$
\widehat\theta_{\text{mle}} = \frac{N_1}{N_0 + N_1}
$$

We see that this is just the empirical fraction of heads, which is an intuitive result.

### Example: MLE for the categorical distribution

Suppose we roll a K-sided dice N times. Let $Y_n \in \{1,...,K\}$ be the n'th outcome, where $Y_n \sim Cat(\theta)$. We want to estimate the probabilities $\theta$ from the dataset $\mathcal{D}= (y_n :n=1:N)$. The NLL is given by

$$
\text{NLL}(\theta)=-\sum_k N_k log(\theta_k)
$$

where $N_k$ is the number of times the event $Y=k$ is observed. (The NLL for the multinomial is the same, up to irrelevant scale factors.)
To compute the MLE, we have to minimize the NLL subject to the constraint that $\sum_{k=1}^{K} \theta_k =1$.  To do this, we will use the method Lagrange multiplies ([Optimization](https://nguyentuss.github.io/p/optimization/)).

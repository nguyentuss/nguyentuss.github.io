---
title: Introduction
description: Welcome to Hugo Theme Stack
date: 2025-03-11 10:17:26+0700
image: cover.jpg
categories:
- Machine_Learning
tags:
- AI
- Machine_Learning
- '#Introduction'
math: true
---

---

## Supervised Learning

The task $T$ is to learn a mapping $f$ from $x \in X$ to $y \in Y$. The $x$ are also called the $\mathbf{features}$. The output $y$ is called the $\mathbf{label}$. The experience $E$ is given in the form of a set of $N$ input-output pairs $\mathcal{D} = \{(x_n,y_n)\},\; n = 1 \rightarrow N$ called the $\textbf{training set}$ (with $N$ as the $\textbf{sample size}$). The performance $P$ depends on the type of output we want to predict.

## Classification

In a classification problem, the output space is a set of $C$ labels called $\textbf{classes}$, $Y = \{1,2,...,C\}$. The problem of predicting the class label given an input is called $\textbf{pattern recognition}$. The goal of supervised learning in a classification problem is to predict the label. A common way to measure performance on this task is by the $\textbf{misclassification rate}$.

$$\mathcal{L}(\boldsymbol{\theta}) \triangleq \frac{1}{N} \sum_{n=1}^{N} \mathbb{I}\left(y_n \neq f(x_n; \boldsymbol{\theta})\right).$$

Here, $\mathbb{I}(e)$ is the indicator function that returns 1 if the condition is true and 0 otherwise. We can also use the notation $\textbf{loss function}$ $l(y,\hat{y})$:

$$\mathcal{L}(\boldsymbol{\theta}) \triangleq \frac{1}{N} \sum_{n=1}^{N} \ell\left(y_n,  f(x_n; \boldsymbol{\theta})\right).$$

## Regression

In regression, the output $y \in \mathbb{R}$ is a real value instead of a discrete label $y \in \{1,...,C\}$. A common choice for the loss function is the quadratic loss, or $\ell_2$ loss:

$$\ell_2(y,\hat{y}) = (y - \hat{y})^2.$$

This penalizes large residuals $y-\hat{y}$. The empirical risk when using quadratic loss is equal to the $\textbf{Mean Squared Error (MSE)}$:

$$MSE(\boldsymbol{\theta})=\frac{1}{N} \sum_{n=1}^{N}(y_n-f(x_n;\boldsymbol{\theta}))^2.$$

![image](img/linear_regression.PNG)

An example of a regression model for 1D data is the $\textbf{linear regression}$ model:

$$f(x;\boldsymbol{\theta})=b+wx.$$

For multiple input features, we can write:

$$f(\mathbf{x};\boldsymbol{\theta})=b+w_1x_1+\cdots+w_Dx_D=b+\mathbf{w}^T\mathbf{x}.$$

![image](img/polynomial_regression.PNG)

We can improve the fit by using a $\textbf{Polynomial regression}$ model with degree $\mathcal{D}$:

$$f(x;\mathbf{w})=\mathbf{w}^T\phi(x),$$

where $\phi(x)=[1,x,x^2,\dots,x^D]$ is the feature vector derived from the input.

## Overfitting

The empirical risk (training loss function) is given by:

$$\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}_{\text{train}}) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x,y}) \in \mathcal{D}_{\text{train}}} \ell(y, f(x; \boldsymbol{\theta})).$$

The difference $\mathcal{L}( \boldsymbol{\theta};p^*)-\mathcal{L}(\boldsymbol{\theta};\mathcal{D}_{\text{train}})$ is called the $\textbf{generalization gap}$. If a model has a large generalization gap (i.e., low empirical risk but high population risk), it is a sign that it is overfitting. In practice we do not know $p^*$, so we partition the data into a training set and a $\textbf{test set}$ to approximate the population risk using the $\textbf{test risk}$:

$$\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}_{\text{test}}) = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x,y}) \in \mathcal{D}_{\text{test}}} \ell(y, f(x; \boldsymbol{\theta})).$$

![image](img/overfitting.PNG)

We can drive the training loss function to zero by increasing the degree $\mathcal{D}$, but this may increase the testing loss function. A model that fits the training data too closely is said to be $\textbf{overfitting}$, while a model that is too simple and does not capture the underlying structure is said to be $\textbf{underfitting}$.

## Unsupervised Learning

In supervised learning, we assume that each input $x$ in the training set has a corresponding target $y$, and our goal is to learn the input-output mapping. In contrast, unsupervised learning deals with data that has no output labels; the dataset is simply $\mathcal{D} = \{x_n : n = 1,\dots,N\}$. Unsupervised learning focuses on modeling the underlying structure of the data by fitting an unconditional model $p(x)$, rather than a conditional model $p(y|x)$.

Unsupervised learning avoids the need for large labeled datasets, which can be time-consuming and expensive to collect, and instead finds patterns, structures, or groupings in the data based on inherent similarities or relationships.

## Clustering

![image](img/clustering.PNG)

A simple example of unsupervised learning is clustering, where the goal is to partition the input data into regions that contain similar points.

## Self-supervised Learning

$\textbf{Self-supervised learning}$ automatically generates $\textbf{labels}$ from $\textbf{unlabeled data}$. For example, one may learn to predict a color image from its grayscale version, or mask out words in a sentence and predict them from the surrounding context. In this setting, a predictor such as

$$\hat{x}_1 = f(x_2;\boldsymbol{\theta})$$

(where $x_2$ is the observed input and $\hat{x}_1$ is the predicted output) learns useful features from the data that can be leveraged in standard supervised tasks.

## Reinforcement Learning

In reinforcement learning, an agent learns how to interact with its environment. For example, a bot playing Mario learns to interact with the game world by moving left or right and jumping when encountering obstacles. ([Click to see the detail](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html))

![image](img/3typeML.PNG)

## Preprocessing Discrete Input Data

### One-hot Encoding

For categorical features, we often convert them into numerical values using $\textbf{one-hot encoding}$. For a categorical variable $x$ with $K$ possible values, one-hot encoding is defined as:

$$\text{one-hot}(x) = [\mathbb{I}(x=1), \dots, \mathbb{I}(x=K)].$$

For example, if $x$ represents one of three colors (red, green, blue), then:

* one-hot(red) = $[1,0,0]$
* one-hot(green) = $[0,1,0]$
* one-hot(blue) = $[0,0,1]$.

### Feature Crosses

To capture interactions between categorical features, we often create composite features. Suppose we want to predict the fuel efficiency of a vehicle based on two categorical variables:

* $x_1$: The type of car (SUV, Truck, Family car).
* $x_2$: The country of origin (USA, Japan).

Using one-hot encoding, we represent these variables as:

$$\phi(x) = [1, I(x_1 = S), I(x_1 = T), I(x_1 = F), I(x_2 = U), I(x_2 = J)].$$

However, this encoding does not capture interactions. To capture interactions between car type and country, we define composite features:

$$\text{(Car type, Country)} = \{(S, U), (T, U), (F, U), (S, J), (T, J), (F, J)\}.$$

The new model becomes:

$$f(x; w) = w^T \phi(x).$$

Expressing this in full:

$$\begin{split}f(x; w) = w_0 + w_1 I(x_1 = S) + w_2 I(x_1 = T) + w_3 I(x_1 = F) \\ + w_4 I(x_2 = U) + w_5 I(x_2 = J) + w_6 I(x_1 = S, x_2 = U) \\ + w_7 I(x_1 = T, x_2 = U) + w_8 I(x_1 = F, x_2 = U) \\ + w_9 I(x_1 = S, x_2 = J) + w_{10} I(x_1 = T, x_2 = J) + w_{11} I(x_1 = F, x_2 = J).\end{split}$$

## Summary

This post introduces key concepts in machine learning—from supervised learning (both classification and regression) to unsupervised, self-supervised, and reinforcement learning. It also covers preprocessing techniques for categorical data such as one-hot encoding and feature crosses. The provided equations are formatted in a single-line style for consistency in this Markdown document.

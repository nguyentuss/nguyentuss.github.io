---
title: index
description: Convolution Neural Network
date: 2025-03-13 21:29:50+0700
image: cover.jpg
categories:
- Deep Learning
tags:
- AI
- '#Introduction'
math: true
---

Convolutional Neural Networks are very similar to ordinary Neural Networks, they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

## Convolution

### Definition

Convolution is a concept in digital signal processing that **transforms** the input information through the **convolution operation** with **a filter** to yield an output in a form of a new signal. This signal will **reduces the features** that the filter is not concerned and just **keep the main features**. Each filter have their main purpose. There are many convolution n-dimension, I will talk about the **2D convolution** because it is the easiest to visualize and also the most common convolution.

The two-dimensional convolution applied to the **2D input matrix** and the **2D filter matrix**. The convolution of an input matrix $X \in \mathbb{R}^{W\times H \times d}$ with a filter $F \in \mathbb{R}^{w\times h \times d}$ produces an output matrix $Y \in \mathbb{R}^{W\times H}$. The steps are follows:

* **Compute the convolution at the single point**: Position the filter at the top-left corner of the input matrix, resulting in a submatrix $X_{sub}$ whose size matches the filter's dimension. The first value,  $y_{11}$ is the convolution of $X_{sub}$ with $F$, such as
  $$
y_{11} = \sum_{}^{w}\sum_{}^{h}\sum_{}^{d} x_{sub}[i,j,u] \times f[i,j,u]
$$
* **Slide the window:** Next, slide the filter window across the input matrix from left to right, and then from top to bottom, using the specified stride. For each position, compute the corresponding output value. Once you have traversed the entire input, you obtain the complete output matrix $Y$. ([Click this link to futher more information about this technique](https://usaco.guide/gold/sliding-window?lang=cpp))

<div style="text-align: center;">
	<img src="img/convlayer_detailedview_demo.gif" width="600">
</div>

In a convolutional neural network (CNN), each subsequent layer takes the output from the layer immediately before it. Therefore, to keep the network design manageable, we need to determine the output size for each layer. This means that given the input (matrix) size $(W_1,H_1)$, a filter of size $(F,F)$, and a stride $S$, we can determine the output matrix $(W_2,H_2)$.
Consider the process sliding with size $1\times W_1$

<div style="text-align: center;">
	<img src="img/stride_convo.png" width="600">
</div>

Assume the process will stop at $W_2$ step. At the first step will reach to position $F$. After each step we will move about $S$, so step $i$ will reach to position $F + (i-1)S$. So that the final step $W_2$ matrix will reach to $F+(W_2-1)S$. This is the highest and closest with $W_1$. In the perfect circumstance the same position $F+(W_2-1)S=W_1$.
$$
W_2=\frac{W_1-F}{S}+1
$$
If there are not in that condition, the division just take the integer, this equation will be
$$
W_2=\lfloor \frac{W_1-F}{S}\rfloor +1
$$
However, we can also make it in the perfect circumstance if we add extra padding on the both edge bound with size $P$ so that the division will divisible by $S$

<div style="text-align: center;">
	<img src="img/conv_padding.png" width="600">
</div>

The equation will be
$$
W_2=\frac{W_1-F+2P}{S}+1
$$
similarly with $H$
$$
H_2=\frac{H_1-F+2P}{S}+1
$$

## Convolution Neural Network

### Definition

In machine learning, a classifier assigns a class label to a data point. For example, an *image classifier* produces a class label (e.g, bird, plane) for what objects exist within an image. A *convolutional neural network*, or CNN for short, is a type of classifier, which excels at solving this problem!
A CNN is a neural network: an algorithm used to recognize patterns in data. Neural Networks in general are composed of a collection of neurons that are organized in layers, each with their own learnable weights and biases. Let’s break down a CNN into its basic building blocks.

1. A **tensor** can be thought of as an n-dimensional matrix. In CNN above, tensors will be 3-dimensional with the exception of the output layer.
1. A **neuron** can be thought of as a function that takes in multiple inputs and yields as a single output. The outputs of neurons  are represented above as the **activation map**.
1. A **layer** are the collection of the neurons in the same operations
1. **Kernel and weights and bias**, while unique to each neuron, are tuned during the training phase, and allow the classifier to adapt to the problem and dataset provided.
1. A CNN conveys a **differentiable score function**, which is represented as a **class score** in the visualization on the output layers.

## Architecture view

**Regular Neural Nets**

## Reference

1. [CNN Explainer](https://poloclub.github.io/cnn-explainer/)(CNN visualization)
1. [Image Kernels explained visually](https://setosa.io/ev/image-kernels/)(Convolution visualization)
1. [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
1. [Khoa học dữ liệu](https://phamdinhkhanh.github.io/2019/08/22/convolutional-neural-network.html)
1. [probml.github.io/pml-book/book1.html](https://probml.github.io/pml-book/book1.html)(book)

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

## Definition

In machine learning, a classifier assigns a class label to a data point. For example, an *image classifier* produces a class label (e.g, bird, plane) for what objects exist within an image. A *convolutional neural network*, or CNN for short, is a type of classifier, which excels at solving this problem!
A CNN is a neural network: an algorithm used to recognize patterns in data. Neural Networks in general are composed of a collection of neurons that are organized in layers, each with their own learnable weights and biases. Let’s break down a CNN into its basic building blocks.

1. A **tensor** can be thought of as an n-dimensional matrix. In CNN above, tensors will be 3-dimensional with the exception of the output layer.
1. A **neuron** can be thought of as a function that takes in multiple inputs and yields as a single output. The outputs of neurons  are represented above as the **activation map**.
1. A **layer** are the collection of the neurons in the same operations
1. **Kernel and weights and bias**, while unique to each neuron, are tuned during the training phase, and allow the classifier to adapt to the problem and dataset provided.
1. A CNN conveys a **differentiable score function**, which is represented as a **class score** in the visualization on the output layers.

## Architecture view

**Regular Neural Nets**

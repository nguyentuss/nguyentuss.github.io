---
title: Optical Flow
description: null
date: 2025-03-13 22:05:35+0700
image: cover.jpg
categories:
- Computer Vision
- Deep Learning
tags:
- CV
- '#Deep_Learning'
- Introduction
math: true
---

Optical flow quantifies the motion of objects between consecutive frames captured by a camera. These algorithms attempt to capture the apparent motion of brightness patterns in the image. It is an important subfield of computer vision, enabling machines to understand scene dynamics and movement.
![](img/pic1.png)

# Basic Gradient-Based Estimation

A common starting point for optical flow estimation is to assume that pixel intensities are translated from one frame to next
$$ I(\vec{x},t)=I(\vec{x}+\vec{u},t+1)$$
where $I(\vec{x},t)$ is an image intensity as a function of space $\vec{x}=(x,y)^T$ and time $t$, and $\vec{u}=(u_1,u_2)^T$ is the 2D velocity. O

# Reference

1. [Video Analysis Algorithms in Computer Vision](https://www.thinkautonomous.ai/blog/computer-vision-from-image-to-video-analysis/)
1. [Thuật toán phân tích video trong thị giác máy tính – VinBigdata Product](https://product.vinbigdata.org/thuat-toan-phan-tich-video-trong-thi-giac-may-tinh/)
1. [Optical Flow Estimation | Papers With Code](https://paperswithcode.com/task/optical-flow-estimation#task-home) (recommend)

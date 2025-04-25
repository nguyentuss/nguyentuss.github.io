---
title: Multivariate Models
description: null
date: 2025-03-10 21:50:35+0700
image: cover.jpg
categories:
- Machine_Learning
tags:
- '#AI'
- '#Machine_Learning'
math: true
---

## Joint distributions for multiple random variables

### Covariance

The **covariance** between two random variables ${X}$ and ${Y}$ measures the **direction** of the **linear relationship** to which ${X}$ and ${Y}$ are (linearly) related. It quantifies how the random variables change together.

*   Positive: If one increases, the other also increases.
*   Negative: If one increases while the other decreases.
*   Zero: There is no relationship between the variables.

$$\textrm{Cov}[X,Y] \triangleq \mathbb{E}\Bigl[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\Bigr] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y].$$

If ${\mathbf{x}}$ is a $D$-dimensional random vector, its **covariance matrix** is defined as

$$\textrm{Cov}[\mathbf{x}] \triangleq \mathbb{E}\Bigl[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{x}-\mathbb{E}[\mathbf{x}])^T\Bigr] \triangleq \mathbf{\Sigma} =.$$

$$\begin{pmatrix} \textrm{Var}[X_1] & \textrm{Cov}[X_1,X_2] & \cdots & \textrm{Cov}[X_1,X_D] \\ \textrm{Cov}[X_2,X_1] & \textrm{Var}[X_2] & \cdots & \textrm{Cov}[X_2,X_D] \\ \vdots & \vdots & \ddots & \vdots \\ \textrm{Cov}[X_D,X_1] & \textrm{Cov}[X_D,X_2] & \cdots & \textrm{Var}[X_D] \end{pmatrix}$$

Covariance itself is the variance of the distribution, from which we can get the important result

$$\mathbb{E}[\mathbf{x}\mathbf{x}^T] = \mathbf{\Sigma} + \mathbf{\mu}\mathbf{\mu}^T.$$

Another useful result is that the covariance of a linear transformation

$$\textrm{Cov}[\mathbf{A}\mathbf{x} + b] = \mathbf{A} \, \textrm{Cov}[\mathbf{x}] \, \mathbf{A}^T.$$

The **cross-covariance** between two random vectors is defined by

$$\textrm{Cov}[\mathbf{x},\mathbf{y}] = \mathbb{E}\Bigl[(\mathbf{x}-\mathbb{E}[\mathbf{x}])(\mathbf{y}-\mathbb{E}[\mathbf{y}])^T\Bigr].$$

### Correlation

![image](img/correlation.png)

Covariance can range over all real numbers. Sometimes it is more convenient to work with a normalized measure that is bounded. The **correlation coefficient** between ${X}$ and ${Y}$ is defined as

$$\rho \triangleq \textrm{Corr}[X,Y] \triangleq \frac{\textrm{Cov}[X,Y]}{\sqrt{\textrm{Var}[X]\textrm{Var}[Y]}}.$$

While covariance can be any real number, correlation is always between ${-1}$ and ${1}$. In the case of a vector ${\mathbf{x}}$ of related variables, the correlation matrix is given by

$$\textrm{Corr}(\mathbf{x}) = \begin{pmatrix} 1 & \frac{\mathbb{E}[(X_1 - \mu_1)(X_2 - \mu_2)]}{\sigma_1 \sigma_2} & \cdots & \frac{\mathbb{E}[(X_1 - \mu_1)(X_D - \mu_D)]}{\sigma_1 \sigma_D} \\ \frac{\mathbb{E}[(X_2 - \mu_2)(X_1 - \mu_1)]}{\sigma_2 \sigma_1} & 1 & \cdots & \frac{\mathbb{E}[(X_2 - \mu_2)(X_D - \mu_D)]}{\sigma_2 \sigma_D} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\mathbb{E}[(X_D - \mu_D)(X_1 - \mu_1)]}{\sigma_D \sigma_1} & \frac{\mathbb{E}[(X_D - \mu_D)(X_2 - \mu_2)]}{\sigma_D \sigma_2} & \cdots & 1 \end{pmatrix}.$$

Note that **uncorrelated does not imply independent**. For example, if ${X}\sim \textrm{Unif}(-1,1)$ and ${Y} = {X}^2$, even though $\textrm{Cov}[X,Y]=0$ and $\textrm{Corr}[X,Y]=0$, ${Y}$ clearly depends on ${X}$.

### Simpson Paradox

![image](img/SimpsonParadox.png)

Simpson's paradox demonstrates that a statistical trend observed in several different groups of data can disappear or even reverse when these groups are combined.

## The Multivariate Gaussian (Normal) Distribution

The multivariate Gaussian (normal) distribution generalizes the univariate Gaussian to multiple dimensions.

$$\mathcal{N}(\mathbf{y};\mathbf{\mu},\mathbf{\Sigma}) \triangleq \frac{1}{(2\pi)^{D/2}|\mathbf{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{y}-\mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{y}-\mathbf{\mu})\right).$$

where ${\mathbf{\mu}} = \mathbb{E}[\mathbf{y}] \in \mathbb{R}^D$ is the mean vector and ${\mathbf{\Sigma}} = \textrm{Cov}[\mathbf{y}]$ is the $D\times D$ covariance matrix.

![image](img/MVN.png)

In 2D, the MVN is known as the **Bivariate Gaussian** distribution. In this case, if ${\mathbf{y}} \sim \mathcal{N}(\mathbf{\mu},\mathbf{\Sigma})$ with

$$\mathbf{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho\,\sigma_1\,\sigma_2 \\ \rho\,\sigma_1\,\sigma_2 & \sigma_2^2 \end{pmatrix},$$

the marginal distribution ${p(y_1)}$ is a 1D Gaussian given by

$$p(y_1) = \mathcal{N}(y_1 \mid \mu_1, \sigma_1^2).$$

and if we observe ${y_2}$, the conditional distribution is

$$p(y_1 \mid y_2) = \mathcal{N}\!\Biggl(y_1 \Bigl| \mu_1 + \frac{\rho\,\sigma_1\,\sigma_2}{\sigma_2^2}(y_2 - \mu_2),\, \sigma_1^2 - \frac{(\rho\,\sigma_1\,\sigma_2)^2}{\sigma_2^2} \Bigr.\Biggr).$$

If $\sigma_1 = \sigma_2 = \sigma$, then

$$p(y_1 \mid y_2) = \mathcal{N}\!\Biggl(y_1 \Bigl| \mu_1 + \rho(y_2 - \mu_2),\, \sigma^2(1 - \rho^2) \Bigr.\Biggr).$$

For instance, if ${\rho = 0.8}$, ${\sigma_1 = \sigma_2 = 1}$, ${\mu_1 = \mu_2 = 0}$, and ${y_2} = 1$, then $\mathbb{E}[y_1 \mid y_2 = 1] = 0.8$ and

$$\textrm{Var}(y_1 \mid y_2 = 1) = 1 - 0.8^2 = 0.36.$$

### Marginals and Conditionals of an MVN

Suppose ${\mathbf{y}} = (y_1, y_2)$ is jointly Gaussian with parameters

$$\mathbf{\mu} = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \quad \mathbf{\Sigma} = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}.$$

Then the marginals are

$$p(y_1) = \mathcal{N}(y_1 \mid \mu_1, \Sigma_{11}), \qquad p(y_2) = \mathcal{N}(y_2 \mid \mu_2, \Sigma_{22}).$$

and the conditional is

$$p(y_1 \mid y_2) = \mathcal{N}(y_1 \mid \mu_{1|2}, \Sigma_{1|2})$$
$$\quad \mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(y_2 - \mu_2), \quad \Sigma_{1|2} = \Sigma_{11} -\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.$$

### Example: Missing Data

Suppose we sample ${N=10}$ vectors from an 8D Gaussian and then hide 50% of the data. For each example ${n}$, let ${v}$ denote the indices of the visible features and ${h}$ the hidden ones. With model parameters ${\theta = (\mathbf{\mu}, \mathbf{\Sigma})}$, we compute the marginal distribution

$$p(\mathbf{y}_{n,h}\mid \mathbf{y}_{n,v}, \theta)$$

for each missing variable ${i \in h}$, and then set the prediction as the posterior mean

$$\bar{y}_{n,i} = \mathbb{E}\!\Bigl[ y_{n,i} \mid \mathbf{y}_{n,v}, \theta \Bigr].$$

The posterior variance

$$\textrm{Var}\!\Bigl[y_{n,i} \mid \mathbf{y}_{n,v}, \theta\Bigr]$$

indicates our confidence in that prediction.

## Linear Gaussian Systems

Consider a scenario where ${z\in \mathbb{R}^L}$ is an unknown value and ${y\in \mathbb{R}^D}$ is a noisy measurement of ${z}$. Assume

$$\begin{aligned} p(z) &= \mathcal{N}(z \mid \mu_z, \Sigma_z), \qquad p(y \mid z) = \mathcal{N}(y \mid Wz + b, \Sigma_y), \end{aligned}$$

where ${W}$ is a ${D\times L}$ matrix. The joint distribution ${p(z,y) = p(z)p(y\mid z)}$ is an $(L+D)$-dimensional Gaussian with mean

$$\mu = \begin{pmatrix} \mu_z \\ W\mu_z + b \end{pmatrix},$$

and covariance

$$\Sigma = \begin{pmatrix} \Sigma_z & \Sigma_zW^T \\ W\Sigma_z & \Sigma_y + W\Sigma_zW^T \end{pmatrix}.$$

### Bayes Rule for Gaussians

The posterior distribution is

$$\begin{aligned} p(z \mid y) &= \mathcal{N}(z \mid \mu_{z \mid y}, \Sigma_{z \mid y}), \quad \Sigma_{z \mid y}^{-1} = \Sigma_z^{-1} + W^T\Sigma_y^{-1}W, \quad \mu_{z \mid y} = \Sigma_{z \mid y}\Bigl[ W^T\Sigma_y^{-1}(y - b) + \Sigma_z^{-1}\mu_z \Bigr]. \end{aligned}$$

The normalization constant is given by

$$\begin{aligned} p(y) &= \int \mathcal{N}(z \mid \mu_z, \Sigma_z)\,\mathcal{N}(y \mid Wz+b, \Sigma_y)\,dz = \mathcal{N}\Bigl(y \mid W\mu_z+b,\, \Sigma_y+W\Sigma_zW^T\Bigr). \end{aligned}$$

### Derivation

The log of the joint distribution is

$$\begin{aligned} \log p(z,y) &= -\frac{1}{2}(z-\mu_z)^T\Sigma_z^{-1}(z-\mu_z) -\frac{1}{2}(y-Wz-b)^T\Sigma_y^{-1}(y-Wz-b). \end{aligned}$$

This quadratic form can be rearranged (by completing the square) to derive the expressions for ${\Sigma_{z \mid y}}$ and ${\mu_{z \mid y}}$.

### Example: Inferring an Unknown Scalar

Suppose we make ${N}$ noisy measurements ${y_i}$ of a scalar ${z}$ with measurement noise precision ${\lambda_y = \frac{1}{\sigma^2}}$:

$$p(y_i \mid z) = \mathcal{N}(y_i \mid z, \lambda_y^{-1}),$$

and assume a prior

$$p(z) = \mathcal{N}(z \mid \mu_0, \lambda_0^{-1}).$$

Then the posterior is

$$p(z \mid y_1,\dots,y_N) = \mathcal{N}(z \mid \mu_N, \lambda_N^{-1}),$$

with

$$\lambda_N = \lambda_0 + N\lambda_y,$$

and

$$\mu_N = \frac{N\lambda_y\overline{y} + \lambda_0\mu_0}{\lambda_N}.$$

In other words, the posterior mean is a weighted average of the prior mean and the sample mean. The posterior variance is

$$\tau_N^2 = \frac{\sigma^2\,\tau_0^2}{N\tau_0^2 + \sigma^2},$$

which decreases as more data is observed.

Sequential updates follow the same principle. After observing ${y_1}$:

$$p(z \mid y_1) = \mathcal{N}(z \mid \mu_1, \sigma_1^2),$$

with

$$\mu_1 = \frac{\sigma_y^2\mu_0 + \sigma_0^2y_1}{\sigma_0^2 + \sigma_y^2}, \quad \sigma_1^2 = \frac{\sigma_0^2\sigma_y^2}{\sigma_0^2 + \sigma_y^2}.$$

Then, using ${p(z \mid y_1)}$ as the new prior, subsequent updates follow similarly.

**Signal-to-noise Ratio (SNR):**

$$\text{SNR} = \frac{\mathbb{E}[Z^2]}{\mathbb{E}[\epsilon^2]} = \frac{\Sigma_0 + \mu_0^2}{\Sigma_y}.$$

This ratio indicates how much the data refines our estimate of ${z}$.

## Mixture Models

---
title: Numerical Method
description: null
date: 2025-03-17 13:18:12+0700
image: cover.jpg
categories:
- Math
- Number Theory
tags:
- '#Math'
math: true
---

## Truncation Errors and the Taylor Series

Truncation errors are those that result from using approximation in place of an exact mathematical procedure.
$$ \frac{dv}{dt} \approx \frac{\Delta v}{\Delta t} = \frac{v(t_{i+1})-v(t_i)}{t_{i+1}-t_i}$$
A truncation error was introduced into the numerical solution because the difference equation only approximates the true value of the derivative. In order to gain insight into the properties of such errors, we now turn to a mathematical formulation that is used widely in numerical methods to express functions in an approximate fashion— the Taylor series.

### The Taylor Series

Taylor’s theorem (Box 4.1) and its associated formula, the Taylor series, is of great value in the study of numerical methods. In essence, the *Taylor series* provides a means to predict a function value at one point in terms of the function value and its derivative at another point. In particular, the theorem states that any smooth function can be approximated as a polynomial.
A useful way to gain insights into Taylor series is to build it term by term. For example, the first term in the series is
$$
f(x_{i+1} \approx f(x_i))
$$
This relationship, called the zero-order approximation, indicates that the value of $f$ at the new point is the same as its value at the old point. This result make intuitive sense because if $x_i$ and $x_{i+1}$ are close to each other, it is likely that the new value is probably similar to the old value.

### Taylor's theorem

If the function $f$ and its $n+1$ derivatives are continuous on an interval containing $a$ and $x$, then the value of the function at $x$ is given by.
$$
f(x) = f(a) +f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+...+\frac{f^{(n)}(a)}{n!}(x-a)^n + R_n
$$
where the **remainder (truncation error)** $R_n$ is defined as
$$
R_n = \int_{a}^{x} \frac{(x-t)^{(n+1)}}{(n+1)!}f^{n+1}
$$

**nth order approximation**
$$
\begin{align}
f(x_{i+1}) = f(x_i) +f'(x_i)(x_{i+1}-x_i)+\frac{f''}{2!}(x_{i+1}-x_i)^2+...\\ +\frac{f^{(n)}}{n!}(x_{i+1}-x_i)^n + R_n
\end{align}
$$
where $(x_{i+1}-x_i)=h$ (step size)

#### Example: Taylor Series Approximation of a Polynomial

**Problem Statement** Use zero- through fourth-order Taylor series expansions to approximate the function
$$ f(x)= -0.1x^4-0.15x^3-0.5x^2-0.25x+1.2$$
from $x_i=0$ with $h=1$. That is, the predict function's value at $x_{i+1}=1$.
![](img/example-taylor-1.png)
**Solution**: Because we are dealing with a known function, we can compute values for $f(x)$ between $0$ and $1$. The results in the figure indicate that the function starts at $f(0)=1.2$ and the curves downward to $f(1)=0.2$. Thus the true value we w to predict is 0.2.

The Taylor series approximation with $n=0$ is
$$ f(x_{i+1}) \approx 1.2 $$

Thus, as in the zero-order approximation is a constant. Using this formulation results in a truncation error of

$$
E_t = 0.2 - 1.2 = -1.0
$$

at $x = 1$.

For $n = 1$, the first derivative must be determined and evaluated at $x = 0$:

$$
f'(0) = -0.4(0.0)^3 - 0.45(0.0)^2 - 1.0(0.0) - 0.25 = -0.25
$$

Therefore, the first-order approximation is

$$
f(x_{i+1}) \approx 1.2 - 0.25h
$$

which can be used to compute $f(1) = 0.95$. Consequently, the approximation begins to capture the downward trajectory of the function in the form of a sloping straight line. This results in a reduction of the truncation error to

$$
E_t = 0.2 - 0.95 = -0.75
$$

For $n = 2$, the second derivative is evaluated at $x = 0$:

$$
f''(0) = -1.2(0.0)^2 - 0.9(0.0) - 1.0 = -1.0
$$

Therefore, according to,

$$
f(x_{i+1}) \approx 1.2 - 0.25h - 0.5h^2
$$

and substituting $h = 1$, $f(1) = 0.45$. The inclusion of the second derivative now adds some downward curvature resulting in an improved estimate, as seen in. The truncation error is reduced further to

$$
0.2 - 0.45 = -0.25.
$$

**Maclaurin Series** is the special case of Taylor series where $a=0$

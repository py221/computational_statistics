# 02 - Distributions

A distribution describes how probability is spread over the values of a random variable. This stage covers
pmf/pdf/cdf, a few common families, moments, and the joint/marginal/conditional machinery for two variables.

## Random variables and distributions

A random variable $X : \Omega \to \mathbb{R}$ induces a distribution through

$$P(X \in S) = P\left(\{\omega \in \Omega \mid X(\omega) \in S\}\right)$$

Example (one coin flip). The event space can be $\mathcal{F} = \{\emptyset, \{H\}, \{T\}, \{H, T\}\}$ and the probability function is
the Bernoulli distribution: $X \sim \text{Bern}(\theta)$ with

$$f(x \mid \theta) = \theta^{x}(1 - \theta)^{1 - x}, \quad x \in \{0, 1\}$$

## Discrete random variables

A discrete random variable has a countable set of possible values. Its **probability mass function** (pmf) $f_X$ satisfies

1. $f_X(x) \ge 0$ for all $x \in \mathbb{R}$
2. $\sum_{x} f_X(x) = 1$
3. $P(a < X \le b) = F_X(b) - F_X(a) = \sum_{a < x \le b} f_X(x)$ for $a < b$

The **cumulative distribution function** (cdf) is

$$F_X(b) = P(X \le b) = \sum_{x \le b} f_X(x)$$

Properties of any cdf:

- non-decreasing: $F_X(x) \le F_X(y)$ whenever $x \le y$
- right-continuous: $\lim_{x \downarrow x_0} F_X(x) = F_X(x_0)$
- takes values in $[0, 1]$, with $F_X(-\infty) = 0$ and $F_X(\infty) = 1$

## Continuous random variables

A continuous random variable takes values in an uncountable set. Its **probability density function** (pdf) $f_X$ satisfies

1. $f_X(x) \ge 0$ for all $x \in \mathbb{R}$ (a density can exceed $1$; only its integral is a probability)
2. $\int_{-\infty}^{\infty} f_X(x)\,dx = 1$
3. $P(a < X \le b) = F_X(b) - F_X(a) = \int_a^b f_X(x)\,dx$ for $a < b$

The cdf and the pdf are linked by

$$F_X(x) = P(X \le x) = \int_{-\infty}^{x} f_X(u)\,du, \qquad f_X(x) = \frac{dF_X(x)}{dx}$$

For a continuous variable $P(X = x) = 0$ for every single point.

## Common distributions

- **Bernoulli**($\theta$): one yes/no trial. $f(x \mid \theta) = \theta^{x}(1-\theta)^{1-x}$, mean $\theta$, variance $\theta(1-\theta)$.
- **Binomial**($n, \theta$): number of successes in $n$ independent Bernoulli trials. $f(k) = \binom{n}{k}\theta^{k}(1-\theta)^{n-k}$, mean $n\theta$, variance $n\theta(1-\theta)$.
- **Poisson**($\lambda$): counts of rare events. $f(k) = e^{-\lambda}\lambda^{k}/k!$, mean and variance both $\lambda$.
- **Exponential**($\lambda$): waiting time. $f(x) = \lambda e^{-\lambda x}$ for $x \ge 0$, mean $1/\lambda$, variance $1/\lambda^{2}$.
- **Normal**($\mu, \sigma^2$): the bell curve, $X \sim \text{Norm}(\mu, \sigma^2)$ with $(x, \mu, \sigma^2) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}_+$ and density given just below.
- **Beta**($a, b$): a distribution on $[0, 1]$, used later as a prior for probabilities. $f(x) \propto x^{a-1}(1-x)^{b-1}$, mean $a/(a+b)$.

The normal density:

$$f_X(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

`distributions.py` plots the pmf/pdf and cdf of these and checks numerically that the cdf is the integral of the pdf.

## Moments, expectations and variances

Moments summarise a distribution, and matching them to data is one way to estimate its parameters (stage 03).

The **expectation** of a function $g(X)$ is its average under the distribution:

$$E[g(X)] = \sum_x g(x)\,p(x) \quad \text{(discrete)}, \qquad E[g(X)] = \int g(x)\,p(x)\,dx \quad \text{(continuous)}$$

- The $r$-th raw moment is $E[X^r]$ (take $g(x) = x^r$). The mean is $E[X]$.
- The $r$-th central moment is $E[(X - E[X])^r]$.

**Variance** is the second central moment:

$$\text{Var}(X) = E\left[(X - E[X])^2\right] = \int (x - E[X])^2\,p(x)\,dx \quad \text{in the continuous case}$$

**Skewness** measures asymmetry:

$$\alpha_3 = \frac{E\left[(X - E[X])^3\right]}{\left(\sqrt{\text{Var}(X)}\right)^{3}}$$

**Kurtosis** measures tail weight ("peakiness"):

$$\alpha_4 = \frac{E\left[(X - E[X])^4\right]}{\left(\sqrt{\text{Var}(X)}\right)^{4}}$$

A normal distribution has $\alpha_3 = 0$ and $\alpha_4 = 3$; "excess kurtosis" is $\alpha_4 - 3$.

Properties:

- $\text{Var}(X) = E[X^2] - (E[X])^2$
- $\text{Var}(c) = 0$ for a constant $c$
- $\text{Var}(aX_1 + bX_2) = a^2\,\text{Var}(X_1) + b^2\,\text{Var}(X_2) + 2ab\,\text{Cov}(X_1, X_2)$, which reduces to $a^2\,\text{Var}(X_1) + b^2\,\text{Var}(X_2)$ when $X_1$ and $X_2$ are uncorrelated (for example independent).
- $E[aX_1 + bX_2] = aE[X_1] + bE[X_2]$ always (linearity).

## Joint distributions and expectations of two variables

When there are two (or more) random variables, possibly with different distributions, the **joint distribution** describes them together:

$$P_{XY}(x, y) = P(X = x, Y = y)$$

and the expectation of a function of both is

$$E[f(X, Y)] = \int\!\!\int f(x, y)\,p(x, y)\,dx\,dy$$

(a double sum in the discrete case).

### Marginal distribution

Given the joint distribution of $X$ and $Y$, the marginal of $X$ sums up the effect of $y$:

$$P_X(x) = \sum_{\text{all } y_i} P_{XY}(x, y_i), \qquad f_X(x) = \int f_{XY}(x, y)\,dy$$

### Conditional distribution

$$P_{X \mid Y}(x_i \mid y_j) = \frac{P_{XY}(x_i, y_j)}{P_Y(y_j)}$$

The conditional expectation is

$$E[X \mid Y = y_j] = \sum_{i} x_i\,P_{X \mid Y}(x_i \mid y_j)$$

or, in the continuous case for a function $g$:

$$E[g(Y) \mid x] = \int_{-\infty}^{\infty} g(y)\,f(y \mid x)\,dy$$

### Independence

$X$ and $Y$ are independent exactly when the joint factorises into a function of $x$ times a function of $y$:

$$f(x, y) = g(x)\,h(y) \quad \text{for all } (x, y)$$

(and then $g$ and $h$ are proportional to the marginals). In the discrete case, a single pair $(x, y)$ that violates $P_{XY}(x, y) = P_X(x)P_Y(y)$ proves the variables are dependent.

### Covariance

$$\text{Cov}(X, Y) = E\left[(X - E[X])(Y - E[Y])\right] = E[XY] - E[X]\,E[Y]$$

### Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X\,\sigma_Y}, \qquad -1 \le \rho_{XY} \le 1$$

Independent variables have $\text{Cov} = 0$, but zero covariance does not imply independence (take $Y = X^2$ with $X$ symmetric around $0$). `joint_covariance.py` shows exactly that.

## Scripts

- `distributions.py` - pmf/pdf/cdf for common families, cdf-equals-integral-of-pdf check, moments computed by numerical integration compared with the closed forms. Saves `output/02_distributions.png`.
- `joint_covariance.py` - a joint table: marginals, conditionals, covariance, correlation, and the zero-covariance-but-dependent example.

# 03 - Estimation: properties, method of moments, MLE, MAP

Given data from a distribution with unknown parameter $\theta$, an **estimator** $\hat{\theta}_n$ is a function of the sample
$x_1, \dots, x_n$. This stage covers how to judge estimators and three ways to construct them.

## Estimator properties

1. **Consistency.** The estimator converges in probability to the true value as the sample grows: for every $\varepsilon > 0$

$$P\left(|\hat{\theta}_n - \theta| > \varepsilon\right) \to 0 \quad \text{as } n \to \infty$$

2. **Bias.** $\text{bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$. The estimator is **unbiased** if $E[\hat{\theta}_n] = \theta$.

3. **Efficiency.** Among unbiased estimators, the efficient one has the lowest possible variance (the Cramer-Rao bound gives the floor).

4. **Mean squared error.**

$$\text{MSE}(\hat{\theta}) = E\left[(\hat{\theta} - \theta)^2\right] = \text{Var}(\hat{\theta}) + \text{bias}(\hat{\theta})^2$$

   MSE captures the trade-off between two kinds of error:
   - **accuracy** is about location: the bias, how far the estimator is centred from the truth
   - **precision** is about spread: the variance, how much the estimator moves from sample to sample

   A little bias can be worth accepting if it buys a larger drop in variance. `estimator_properties.py` shows this for the variance of a normal sample: dividing by $n+1$ is biased but has lower MSE than the unbiased divisor $n-1$.

## Method of moments (MoM)

Match population moments to sample moments. The $k$-th population moment $E[X^k]$ is replaced with its sample counterpart:

$$E[X^k] = \int x^k\,p(x)\,dx \;\approx\; \frac{1}{n}\sum_{i=1}^{n} x_i^k$$

With $k = 1$ this is $\mu \approx \bar{x}$. The number of moments you need equals the number of parameters to estimate.

### Example: Bernoulli

$X_i \sim \text{Bern}(\theta)$ with pmf $f(x \mid \theta) = \theta^{x}(1-\theta)^{1-x}$.

Data: $n = 20$ flips, $\{1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,0,1,1,1,0\}$, which contains $13$ heads.

Population moments:

$$E[X] = \sum_x x\,f(x) = 0\cdot(1-\theta) + 1\cdot\theta = \theta, \qquad \text{Var}(X) = \theta(1-\theta)$$

Matching the first sample moment to the first population moment:

$$\hat{\theta}_{MoM} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{13}{20} = 0.65$$

### Example: Gamma (where MoM and MLE disagree)

For $\text{Gamma}(k, s)$ with shape $k$ and scale $s$: $E[X] = ks$ and $\text{Var}(X) = ks^2$. Matching the sample mean $\bar{x}$ and sample variance $v$ gives

$$\hat{s} = \frac{v}{\bar{x}}, \qquad \hat{k} = \frac{\bar{x}^2}{v}$$

MoM is quick and closed-form but usually less efficient than MLE. `mom_vs_mle.py` compares the two.

## Maximum likelihood estimation (MLE)

**Assumption.** The data are independent and identically distributed (iid) draws from a population with density/pmf $f(x \mid \theta)$.

**Goal.** Find the $\theta$ that makes the observed data most likely.

**Likelihood function.**

$$\mathcal{L}(\theta \mid x_1, \dots, x_n) = \prod_{i=1}^{n} f(x_i \mid \theta)$$

Solve $\frac{d}{d\theta}\mathcal{L}(\theta \mid x) = 0$, then check that the solution is a maximum and not a boundary point.

**Log-likelihood.** Maximising $\ln \mathcal{L}$ gives the same $\hat{\theta}$ and is easier: products become sums and numerical underflow is avoided. For the Bernoulli:

$$\ln \mathcal{L}(\theta \mid x_1, \dots, x_n) = \left(\sum_{i=1}^{n} x_i\right)\ln\theta + \left(\sum_{i=1}^{n}(1 - x_i)\right)\ln(1-\theta)$$

Differentiating with respect to $\theta$ and setting to zero:

$$\left(\sum_{i=1}^{n} x_i\right)\frac{1}{\hat{\theta}} - \left(\sum_{i=1}^{n}(1 - x_i)\right)\frac{1}{1-\hat{\theta}} = 0$$

Multiplying through by $\hat{\theta}(1-\hat{\theta})$ and solving gives

$$\hat{\theta}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

The second derivative is negative on $(0,1)$, so this is a maximum. For the data above, $\hat{\theta}_{MLE} = 13/20$, the same as MoM.

### Example: Normal

For $X_i \sim \text{Norm}(\mu, \sigma^2)$ the MLEs are

$$\hat{\mu} = \bar{x}, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

The variance MLE divides by $n$, so it is biased (by a factor $(n-1)/n$) but consistent.

## Maximum a posteriori (MAP) estimation

MAP is MLE augmented with a **prior** $\pi(\theta)$: the belief we hold about the parameter before seeing the data.

$$\hat{\theta}_{MAP} = \arg\max_{\theta}\; \mathcal{L}(\theta \mid x_1, \dots, x_n)\,\pi(\theta)$$

The prior $\pi(\theta)$ is the additional information. A flat prior makes MAP equal to MLE; a strong prior pulls the estimate towards what we believed beforehand.

### Example: Bernoulli with a Beta prior

With $\pi(\theta) = \text{Beta}(a, b)$, i.e. $\pi(\theta) \propto \theta^{a-1}(1-\theta)^{b-1}$, and $s = \sum x_i$ successes in $n$ trials:

$$\hat{\theta}_{MAP} = \frac{s + a - 1}{n + a + b - 2}$$

For $13$ heads in $20$ flips and a $\text{Beta}(2, 2)$ prior: $\hat{\theta}_{MAP} = 14/22 \approx 0.636$, pulled slightly towards $0.5$ relative to $0.65$. $\text{Beta}(1, 1)$ (uniform) gives back the MLE. Stage 05 develops this idea.

## Scripts

- `bernoulli_mle_map.py` - closed-form versus numerical MLE and MAP for the 20-flip example, with a plot of likelihood and posterior.
- `mom_vs_mle.py` - MoM versus MLE for a Gamma sample.
- `estimator_properties.py` - simulation of bias, variance, MSE and consistency.

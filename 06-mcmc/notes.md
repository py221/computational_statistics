# 06 - Bayesian inference with MCMC

Conjugate priors (stage 05) give closed-form posteriors only for a short list of model pairs. For everything else we need to
work with $\pi(\theta \mid x) \propto \mathcal{L}(\theta \mid x)\,\pi(\theta)$ without knowing its normalising constant.
Markov chain Monte Carlo (MCMC) does that: it builds a chain of parameter values whose long-run distribution is the posterior.

## Why sampling

If we can draw samples $\theta^{(1)}, \dots, \theta^{(T)}$ from the posterior, any posterior quantity becomes an average:

$$E[g(\theta) \mid x] \approx \frac{1}{T}\sum_{t=1}^{T} g\left(\theta^{(t)}\right)$$

Means, quantiles, credible intervals and predictive probabilities all follow. The challenge is drawing the samples when the
normalising constant $P(x) = \int \mathcal{L}(\theta \mid x)\,\pi(\theta)\,d\theta$ is intractable.

## Markov chains in one paragraph

A Markov chain moves from $\theta^{(t)}$ to $\theta^{(t+1)}$ using a rule that depends only on the current state. Under mild conditions
(irreducible, aperiodic) the chain has a unique **stationary distribution** it converges to, regardless of the starting point. MCMC designs the rule so that this stationary distribution is the posterior.

## Metropolis-Hastings

To target a density $p(\theta) \propto \tilde{p}(\theta)$ known only up to a constant:

1. Start at $\theta^{(0)}$.
2. Propose $\theta^{*} \sim q(\theta^{*} \mid \theta^{(t)})$.
3. Compute the acceptance probability

$$\alpha = \min\left(1,\; \frac{\tilde{p}(\theta^{*})\,q(\theta^{(t)} \mid \theta^{*})}{\tilde{p}(\theta^{(t)})\,q(\theta^{*} \mid \theta^{(t)})}\right)$$

4. With probability $\alpha$ set $\theta^{(t+1)} = \theta^{*}$, otherwise keep $\theta^{(t+1)} = \theta^{(t)}$ (a rejected move still counts as a sample).

The normalising constant cancels in the ratio, which is the point. For a **symmetric** proposal such as the random walk
$\theta^{*} = \theta^{(t)} + \varepsilon$, $\varepsilon \sim \text{Norm}(0, s^2)$, the $q$ terms cancel too (the Metropolis algorithm), and for a posterior

$$\alpha = \min\left(1,\; \frac{\mathcal{L}(\theta^{*} \mid x)\,\pi(\theta^{*})}{\mathcal{L}(\theta^{(t)} \mid x)\,\pi(\theta^{(t)})}\right)$$

In practice we work with log densities and compare $\ln u < \ln\tilde{p}(\theta^{*}) - \ln\tilde{p}(\theta^{(t)})$ for $u \sim \text{Unif}(0,1)$, to avoid underflow.

### Tuning the step size

- **Too small**: nearly every proposal is accepted, but the chain barely moves, so successive samples are highly correlated.
- **Too large**: most proposals land in low-density regions and are rejected, so the chain stays put.
- A common rule of thumb is an acceptance rate of roughly $0.44$ in one dimension and $0.23$ in many dimensions.

`mh_beta_binomial.py` runs all three regimes and compares the good one with the exact conjugate posterior.

## Diagnostics

An MCMC sample is only trustworthy if the chain has converged and mixes well. Run several chains from dispersed starting points and check:

1. **Trace plots.** Should look like a stationary "fuzzy caterpillar" with chains overlapping. Drifting, plateaus or separated chains are warning signs.
2. **Burn-in (warm-up).** Early samples still reflect the arbitrary starting point and are discarded.
3. **Acceptance rate.** For random-walk Metropolis, far from the ranges above suggests the step size needs tuning.
4. **Autocorrelation and effective sample size (ESS).** Successive draws are correlated, so $T$ draws are worth fewer independent ones:

$$\text{ESS} = \frac{T}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$

   where $\rho_k$ is the lag-$k$ autocorrelation (the sum is truncated once the autocorrelations become noise). The Monte Carlo standard error of a posterior mean is about $\text{sd}/\sqrt{\text{ESS}}$.
5. **R-hat (potential scale reduction).** Compares between-chain and within-chain variance. With $m$ chains of length $n$, within-chain variance $W$ and between-chain variance $B$:

$$\hat{R} = \sqrt{\frac{\frac{n-1}{n}W + \frac{1}{n}B}{W}}$$

   Values close to $1$ (below about $1.01$ to $1.05$) indicate the chains agree. This repo uses the *split* version, which also splits each chain in half to catch within-chain drift.

These are basic checks; none of them proves convergence, they only fail to find problems.

## Scripts

- `mcmc.py` - a random-walk Metropolis sampler in numpy plus `autocorrelation`, `effective_sample_size` and `split_rhat`.
- `mh_beta_binomial.py` - samples the Beta-Binomial posterior from stage 05 with several step sizes and chains, prints acceptance, ESS, R-hat and compares with the exact answer. Saves `output/06_trace.png` and `output/06_posterior.png`.

# 07 - PyMC

Stage 06 wrote a sampler by hand to see how MCMC works. In practice you describe the model and let a library do the sampling.
[PyMC](https://www.pymc.io) is a probabilistic programming library: you declare priors and a likelihood, and it samples the posterior
with the No-U-Turn Sampler (NUTS), an adaptive Hamiltonian Monte Carlo method that uses gradients and needs no step-size tuning by hand.
[ArviZ](https://python.arviz.org) provides the summaries and diagnostics.

## A model is the same Bayes' rule

The Beta-Binomial model from stages 04-06 in PyMC:

```python
import pymc as pm

with pm.Model():
    theta = pm.Beta("theta", alpha=2, beta=2)                 # prior
    pm.Binomial("heads", n=20, p=theta, observed=13)           # likelihood
    idata = pm.sample(draws=2000, tune=1000, chains=4)         # posterior samples
```

The model block states $\theta \sim \text{Beta}(2, 2)$ and $\text{heads} \mid \theta \sim \text{Binomial}(20, \theta)$. Sampling returns draws from

$$\pi(\theta \mid \text{heads}) \propto \theta^{13}(1-\theta)^{7}\cdot\theta(1-\theta)$$

which is the $\text{Beta}(15, 9)$ distribution from stage 05, so there is an exact answer to compare against.

## What to check

- `az.summary(idata)` reports the posterior mean, standard deviation, credible interval, ESS and R-hat. These are the same diagnostics as in stage 06, computed for you.
- Divergences, reported during sampling, signal regions the sampler could not explore reliably.
- Plots such as `az.plot_trace` mirror the trace plots from stage 06.

## A second model: linear regression

`linear_regression_pymc.py` fits $y = \alpha + \beta x + \varepsilon$, $\varepsilon \sim \text{Norm}(0, \sigma^2)$ with weakly informative priors on
$\alpha$, $\beta$ and $\sigma$, and compares the posterior means with ordinary least squares. With weak priors and a decent amount of data they should be close.

## Scripts

- `beta_binomial_pymc.py` - the model above, compared with the exact conjugate posterior and the hand-written Metropolis sampler.
- `linear_regression_pymc.py` - Bayesian linear regression versus least squares.

Both scripts print a clear message and exit if PyMC or ArviZ is not installed. Install them with `pip install -r requirements.txt`.
PyMC compiles its model with a C compiler through PyTensor; on a machine without one it falls back to slower Python code, which is still fine for these small models.

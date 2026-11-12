"""Bayesian linear regression in PyMC versus ordinary least squares.

Run:  python 07-pymc/linear_regression_pymc.py
"""
import sys

try:
    import arviz as az
    import pymc as pm
except ImportError as exc:  # pragma: no cover - depends on the environment
    print(f"PyMC/ArviZ are not installed ({exc}). Install them with: pip install -r requirements.txt")
    sys.exit(0)

import numpy as np

TRUE_ALPHA, TRUE_BETA, TRUE_SIGMA = 1.0, 2.5, 0.8


def make_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = TRUE_ALPHA + TRUE_BETA * x + rng.normal(0, TRUE_SIGMA, n)
    return x, y


def ols(x, y):
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def fit(x, y, draws=1500, tune=1000, chains=4, seed=0):
    with pm.Model():
        alpha = pm.Normal("alpha", 0, 10)
        beta = pm.Normal("beta", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", mu=alpha + beta * x, sigma=sigma, observed=y)
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                         progressbar=False)


def main():
    x, y = make_data()
    idata = fit(x, y)
    print(az.summary(idata, var_names=["alpha", "beta", "sigma"]))
    a_ols, b_ols = ols(x, y)
    post = idata.posterior
    print(f"\nOLS:       alpha={a_ols:.3f}  beta={b_ols:.3f}")
    print(f"Posterior: alpha={float(post['alpha'].mean()):.3f}  beta={float(post['beta'].mean()):.3f}")
    print(f"Truth:     alpha={TRUE_ALPHA}  beta={TRUE_BETA}  sigma={TRUE_SIGMA}")


if __name__ == "__main__":
    main()

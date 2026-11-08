"""Conjugate updates: Beta-Binomial and Normal-Normal (known variance).

Run:  python 05-priors/conjugate.py   (demo and brute-force check)
"""
import numpy as np
from scipy import stats


def beta_binomial_update(a, b, successes, n):
    """Posterior Beta(a + s, b + n - s) for a Beta(a, b) prior and s successes in n trials."""
    return a + successes, b + n - successes


def beta_summary(a, b):
    dist = stats.beta(a, b)
    mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else float("nan")
    lo, hi = dist.interval(0.95)
    return {"mean": float(dist.mean()), "mode": float(mode), "sd": float(dist.std()), "ci95": (float(lo), float(hi))}


def normal_normal_update(mu0, tau0, x, sigma):
    """Posterior (mean, sd) of mu given data x ~ N(mu, sigma^2), prior mu ~ N(mu0, tau0^2)."""
    x = np.asarray(x, dtype=float)
    precision = 1.0 / tau0**2 + x.size / sigma**2
    mean = (mu0 / tau0**2 + x.sum() / sigma**2) / precision
    return mean, precision**-0.5


def grid_posterior_normal(mu0, tau0, x, sigma, grid):
    """Brute-force check: normalise likelihood * prior on a grid of mu values."""
    x = np.asarray(x, dtype=float)
    log_post = stats.norm.logpdf(grid, mu0, tau0)
    for xi in x:
        log_post = log_post + stats.norm.logpdf(xi, grid, sigma)
    post = np.exp(log_post - log_post.max())
    return post / np.trapezoid(post, grid)


def main():
    a, b = beta_binomial_update(2, 2, successes=13, n=20)
    print(f"Beta(2,2) prior + 13/20 -> Beta({a},{b})", beta_summary(a, b))

    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=25)
    mean, sd = normal_normal_update(mu0=0.0, tau0=10.0, x=x, sigma=2.0)
    grid = np.linspace(mean - 6 * sd, mean + 6 * sd, 4001)
    post = grid_posterior_normal(0.0, 10.0, x, 2.0, grid)
    grid_mean = np.trapezoid(grid * post, grid)
    grid_sd = np.sqrt(np.trapezoid((grid - grid_mean) ** 2 * post, grid))
    print(f"Normal-Normal analytic: mean={mean:.4f} sd={sd:.4f};  grid: mean={grid_mean:.4f} sd={grid_sd:.4f}")


if __name__ == "__main__":
    main()

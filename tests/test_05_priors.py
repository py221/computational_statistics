import numpy as np
import pytest
from scipy import stats

import conjugate as cj
import prior_sensitivity as psens


def test_beta_binomial_update():
    assert cj.beta_binomial_update(2, 2, 13, 20) == (15, 9)


def test_beta_binomial_matches_grid():
    a, b = cj.beta_binomial_update(3, 5, 13, 20)
    theta = np.linspace(1e-6, 1 - 1e-6, 20001)
    unnorm = theta**13 * (1 - theta) ** 7 * stats.beta(3, 5).pdf(theta)
    post = unnorm / np.trapezoid(unnorm, theta)
    assert np.abs(post - stats.beta(a, b).pdf(theta)).max() < 1e-3


def test_normal_normal_matches_grid():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, size=30)
    mean, sd = cj.normal_normal_update(1.0, 3.0, x, 2.0)
    grid = np.linspace(mean - 8 * sd, mean + 8 * sd, 8001)
    post = cj.grid_posterior_normal(1.0, 3.0, x, 2.0, grid)
    gm = np.trapezoid(grid * post, grid)
    gs = np.sqrt(np.trapezoid((grid - gm) ** 2 * post, grid))
    assert (mean, sd) == pytest.approx((gm, gs), rel=1e-4)


def test_posterior_mean_is_weighted_average():
    a, b, s, n = 4, 6, 13, 20
    w = (a + b) / (a + b + n)
    expected = w * a / (a + b) + (1 - w) * s / n
    pa, pb = cj.beta_binomial_update(a, b, s, n)
    assert pa / (pa + pb) == pytest.approx(expected)


def test_prior_matters_less_with_more_data():
    small = psens.posterior_means(13, 20)
    large = psens.posterior_means(1300, 2000)
    spread_small = max(small.values()) - min(small.values())
    spread_large = max(large.values()) - min(large.values())
    assert spread_large < spread_small / 5

import numpy as np
import pytest
from scipy import stats

import bayes_rule as br
import decision_boundary as db
import posterior_summaries as ps


def test_diagnostic_test_posterior():
    assert br.posterior_disease(0.01, 0.95, 0.05) == pytest.approx(0.95 * 0.01 / (0.95 * 0.01 + 0.05 * 0.99))


def test_grid_posterior_matches_beta():
    theta, post = br.grid_posterior(13, 20, lambda t: np.ones_like(t))
    exact = stats.beta(14, 8)
    assert np.trapezoid(post, theta) == pytest.approx(1.0)
    assert np.abs(post - exact.pdf(theta)).max() < 1e-3


def test_equal_cost_boundary_minimises_error():
    grid = np.linspace(-2, 4, 3001)
    best = grid[np.argmin([db.error_rates(x)[2] for x in grid])]
    assert db.boundary(0.5) == pytest.approx(best, abs=2e-3)


def test_cost_threshold():
    assert db.threshold_from_costs(3, 1) == pytest.approx(0.75)
    assert db.boundary(0.75) > db.boundary(0.5)


def test_loss_minimisers_are_mean_median_mode():
    mean, median, mode = ps.best_action_by_loss()
    p = ps.POSTERIOR
    assert mean == pytest.approx(p.mean(), abs=1e-3)
    assert median == pytest.approx(p.median(), abs=1e-3)
    a, b = p.args
    assert mode == pytest.approx((a - 1) / (a + b - 2), abs=2e-3)

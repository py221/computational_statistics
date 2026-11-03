import numpy as np
import pytest

import bernoulli_mle_map as bm
import estimator_properties as ep
import mom_vs_mle as mm


def test_bernoulli_mle_matches_closed_form():
    x = bm.FLIPS
    assert x.sum() == 13
    assert bm.mle_closed_form(x) == pytest.approx(13 / 20)
    assert bm.mle_numeric(x) == pytest.approx(13 / 20, abs=1e-6)
    assert bm.mom_closed_form(x) == bm.mle_closed_form(x)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 2), (10, 10), (8, 2)])
def test_map_matches_closed_form(a, b):
    x = bm.FLIPS
    assert bm.map_numeric(x, a, b) == pytest.approx(bm.map_closed_form(x, a, b), abs=1e-6)


def test_uniform_prior_map_equals_mle():
    assert bm.map_closed_form(bm.FLIPS, 1, 1) == pytest.approx(bm.mle_closed_form(bm.FLIPS))


def test_normal_mle_variance_divides_by_n():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, size=500)
    assert x.var() == pytest.approx(((x - x.mean()) ** 2).sum() / x.size)


def test_gamma_mom_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.gamma(3.0, 2.0, size=20_000)
    k, s = mm.gamma_mom(x)
    assert (k, s) == pytest.approx((3.0, 2.0), rel=0.05)
    k, s = mm.gamma_mle(x)
    assert (k, s) == pytest.approx((3.0, 2.0), rel=0.05)


def test_mse_is_variance_plus_bias_squared():
    for bias, var, mse in ep.variance_estimators(reps=50_000).values():
        assert mse == pytest.approx(var + bias**2, rel=1e-6)


def test_bias_ordering_and_mse_tradeoff():
    res = ep.variance_estimators(reps=100_000)
    assert abs(res["n-1 (unbiased)"][0]) < 0.05
    assert res["n+1 (min MSE)"][2] < res["n-1 (unbiased)"][2]


def test_consistency():
    probs = ep.consistency(sizes=(10, 1000), reps=1000)
    assert probs[1000] < probs[10]
    assert probs[1000] < 0.01

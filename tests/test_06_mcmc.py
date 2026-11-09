import numpy as np
import pytest
from scipy import stats

import mcmc
import mh_beta_binomial as mh


def test_mh_posterior_mean_close_to_conjugate():
    draws, rates = mh.sample(step=0.2, n_samples=8000, seed=1)
    exact = stats.beta(15, 9)
    assert draws.mean() == pytest.approx(exact.mean(), abs=0.01)
    assert draws.std() == pytest.approx(exact.std(), abs=0.01)
    assert 0.2 < rates.mean() < 0.8


def test_mh_recovers_standard_normal():
    rng = np.random.default_rng(0)
    s, rate = mcmc.metropolis(lambda x: -0.5 * float(x @ x), [0.0, 0.0], 40_000, 1.0, rng)
    assert s[2000:].mean(axis=0) == pytest.approx([0, 0], abs=0.1)
    assert s[2000:].std(axis=0) == pytest.approx([1, 1], abs=0.1)


def test_rhat_flags_non_mixing_chains():
    good, _ = mh.sample(step=0.2)
    assert mcmc.split_rhat(good) < 1.02
    rng = np.random.default_rng(0)
    separated = np.stack([rng.normal(0, 1, 1000), rng.normal(5, 1, 1000)])
    assert mcmc.split_rhat(separated) > 1.5


def test_ess_of_iid_and_correlated_series():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=5000)
    assert mcmc.effective_sample_size(iid) > 0.7 * iid.size
    ar = np.zeros(5000)
    for t in range(1, ar.size):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    # theoretical ESS for AR(1) with phi=.9 is n*(1-phi)/(1+phi) ~ 0.053 n
    assert mcmc.effective_sample_size(ar) < 0.15 * ar.size


def test_small_step_has_lower_ess_than_good_step():
    small, _ = mh.sample(step=0.005)
    good, _ = mh.sample(step=0.2)
    assert mh.summarise(small)["ess"] < mh.summarise(good)["ess"] / 10


def test_autocorrelation_lag_zero_is_one():
    x = np.random.default_rng(0).normal(size=100)
    assert mcmc.autocorrelation(x, 5)[0] == pytest.approx(1.0)

import numpy as np
import pytest
from scipy import stats

import distributions as dist
import joint_covariance as jc


@pytest.mark.parametrize("d", [stats.norm(0, 1), stats.expon(), stats.beta(2, 5)])
def test_cdf_is_integral_of_pdf(d):
    a, b = dist.cdf_vs_integral(d, 0.7)
    assert a == pytest.approx(b, abs=1e-6)


def test_numeric_moments_match_closed_form():
    mean, var, skew, kurt = dist.numeric_moments(stats.expon(), lower=0, upper=60)
    assert (mean, var) == pytest.approx((1.0, 1.0), abs=1e-4)
    assert (skew, kurt) == pytest.approx((2.0, 9.0), abs=1e-3)
    mean, var, skew, kurt = dist.numeric_moments(stats.norm(), lower=-12, upper=12)
    assert (mean, var, skew, kurt) == pytest.approx((0.0, 1.0, 0.0, 3.0), abs=1e-4)


def test_pmf_sums_to_one():
    ks = np.arange(0, 11)
    assert stats.binom(10, 0.3).pmf(ks).sum() == pytest.approx(1.0)


def test_joint_marginals_and_covariance():
    px, py = jc.marginals(jc.JOINT)
    assert px.sum() == pytest.approx(1.0) and py.sum() == pytest.approx(1.0)
    # covariance matches sampling from the joint table
    rng = np.random.default_rng(0)
    idx = rng.choice(jc.JOINT.size, p=jc.JOINT.ravel(), size=400_000)
    xs = jc.X_VALUES[idx // jc.JOINT.shape[1]]
    ys = jc.Y_VALUES[idx % jc.JOINT.shape[1]]
    assert np.cov(xs, ys, bias=True)[0, 1] == pytest.approx(jc.covariance(jc.JOINT, jc.X_VALUES, jc.Y_VALUES), abs=5e-3)
    assert abs(jc.correlation(jc.JOINT, jc.X_VALUES, jc.Y_VALUES)) <= 1


def test_zero_covariance_does_not_imply_independence():
    joint, xs, ys = jc.zero_cov_but_dependent()
    assert jc.covariance(joint, xs, ys) == pytest.approx(0.0, abs=1e-12)
    assert not jc.is_independent(joint)

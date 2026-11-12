import sys
from pathlib import Path

import pytest
from scipy import stats

pytest.importorskip("pymc")
pytest.importorskip("arviz")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07-pymc"))
import beta_binomial_pymc as bb  # noqa: E402
import linear_regression_pymc as lr  # noqa: E402


def test_pymc_beta_binomial_matches_exact():
    idata = bb.fit(draws=1000, tune=500, chains=2)
    draws = idata.posterior["theta"].values.ravel()
    exact = stats.beta(15, 9)
    assert draws.mean() == pytest.approx(exact.mean(), abs=0.02)
    assert draws.std() == pytest.approx(exact.std(), abs=0.02)


def test_pymc_regression_close_to_ols():
    x, y = lr.make_data()
    idata = lr.fit(x, y, draws=800, tune=500, chains=2)
    a_ols, b_ols = lr.ols(x, y)
    post = idata.posterior
    assert float(post["alpha"].mean()) == pytest.approx(a_ols, abs=0.1)
    assert float(post["beta"].mean()) == pytest.approx(b_ols, abs=0.1)

"""Beta-Binomial in PyMC, compared with the exact conjugate posterior and hand-written Metropolis.

Run:  python 07-pymc/beta_binomial_pymc.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import arviz as az
    import pymc as pm
except ImportError as exc:  # pragma: no cover - depends on the environment
    print(f"PyMC/ArviZ are not installed ({exc}). Install them with: pip install -r requirements.txt")
    sys.exit(0)

import numpy as np
from scipy import stats

sys.path.insert(0, str(ROOT / "05-priors"))
sys.path.insert(0, str(ROOT / "06-mcmc"))
from conjugate import beta_binomial_update  # noqa: E402
from mh_beta_binomial import sample as mh_sample  # noqa: E402

HEADS, N = 13, 20
PRIOR = (2.0, 2.0)


def fit(draws=2000, tune=1000, chains=4, seed=0):
    with pm.Model():
        theta = pm.Beta("theta", alpha=PRIOR[0], beta=PRIOR[1])
        pm.Binomial("heads", n=N, p=theta, observed=HEADS)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=False)
    return idata


def main():
    idata = fit()
    print(az.summary(idata, var_names=["theta"]))

    pymc_draws = idata.posterior["theta"].values.ravel()
    a, b = beta_binomial_update(*PRIOR, HEADS, N)
    exact = stats.beta(a, b)
    mh_draws, _ = mh_sample(step=0.2)
    mh_draws = mh_draws.ravel()

    print(f"\n{'method':12s} {'mean':>8s} {'sd':>8s} {'2.5%':>8s} {'97.5%':>8s}")
    rows = {
        "exact": (exact.mean(), exact.std(), exact.ppf(0.025), exact.ppf(0.975)),
        "PyMC NUTS": (pymc_draws.mean(), pymc_draws.std(), *np.quantile(pymc_draws, [0.025, 0.975])),
        "Metropolis": (mh_draws.mean(), mh_draws.std(), *np.quantile(mh_draws, [0.025, 0.975])),
    }
    for name, r in rows.items():
        print(f"{name:12s} " + " ".join(f"{v:8.4f}" for v in r))


if __name__ == "__main__":
    main()

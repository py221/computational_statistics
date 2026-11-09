"""Metropolis-Hastings on the Beta-Binomial posterior, checked against the exact conjugate answer.

Run:  python 06-mcmc/mh_beta_binomial.py
Saves output/06_trace.png and output/06_posterior.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcmc import effective_sample_size, run_chains, split_rhat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "05-priors"))
from conjugate import beta_binomial_update  # noqa: E402

OUT = ROOT / "output"

HEADS, N = 13, 20
PRIOR = (2.0, 2.0)


def log_posterior(theta, heads=HEADS, n=N, prior=PRIOR):
    """Unnormalised log posterior: log-likelihood + log-prior. Zero outside (0, 1)."""
    t = float(theta[0])
    if not 0.0 < t < 1.0:
        return -np.inf
    a, b = prior
    return (heads + a - 1) * np.log(t) + (n - heads + b - 1) * np.log1p(-t)


def sample(step, n_samples=6000, burn=1000, starts=(0.1, 0.5, 0.9, 0.3), seed=0):
    chains, rates = run_chains(log_posterior, [[s] for s in starts], n_samples, step, seed=seed)
    return chains[:, burn:, 0], rates


def summarise(draws):
    return {
        "mean": float(draws.mean()),
        "ess": float(np.mean([effective_sample_size(c) for c in draws])),
        "rhat": split_rhat(draws),
    }


def main():
    OUT.mkdir(exist_ok=True)
    a, b = beta_binomial_update(*PRIOR, HEADS, N)
    exact = stats.beta(a, b)
    print(f"exact posterior Beta({a:g},{b:g}): mean={exact.mean():.4f} sd={exact.std():.4f}\n")

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    results = {}
    for ax, (label, step) in zip(axes, [("too small", 0.005), ("reasonable", 0.2), ("too large", 3.0)]):
        draws, rates = sample(step)
        s = summarise(draws)
        results[label] = (draws, s)
        print(f"step={step:<6} ({label:10s}) acceptance={rates.mean():.2f}  mean={s['mean']:.4f}  "
              f"ESS/chain={s['ess']:.0f}  R-hat={s['rhat']:.3f}")
        for c in draws:
            ax.plot(c[:1500], lw=0.6)
        ax.set_ylabel(f"step {step}")
        ax.set_title(label, fontsize=9, loc="left")
    axes[-1].set_xlabel("iteration (after burn-in)")
    fig.tight_layout()
    fig.savefig(OUT / "06_trace.png", dpi=110)
    plt.close(fig)

    draws = results["reasonable"][0].ravel()
    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(draws, bins=60, density=True, alpha=0.5, label="Metropolis samples")
    ax.plot(x, exact.pdf(x), "k", label=f"exact Beta({a:g},{b:g})")
    ax.legend()
    ax.set_xlabel(r"$\theta$")
    fig.tight_layout()
    fig.savefig(OUT / "06_posterior.png", dpi=110)
    plt.close(fig)

    print(f"\nMH posterior mean={draws.mean():.4f} vs exact {exact.mean():.4f}; "
          f"95% interval MH=({np.quantile(draws, .025):.3f}, {np.quantile(draws, .975):.3f}) "
          f"exact=({exact.ppf(.025):.3f}, {exact.ppf(.975):.3f})")


if __name__ == "__main__":
    main()

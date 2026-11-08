"""Prior sensitivity: the same proportion of heads under several Beta priors, small vs large n.

Run:  python 05-priors/prior_sensitivity.py
Saves output/05_prior_sensitivity.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from conjugate import beta_binomial_update

OUT = Path(__file__).resolve().parents[1] / "output"

PRIORS = {
    "flat Beta(1,1)": (1, 1),
    "Jeffreys Beta(.5,.5)": (0.5, 0.5),
    "weak Beta(2,2)": (2, 2),
    "informative Beta(40,60)": (40, 60),
    "misleading Beta(80,20)": (80, 20),
}


def posterior_means(successes, n):
    return {name: stats.beta(*beta_binomial_update(a, b, successes, n)).mean() for name, (a, b) in PRIORS.items()}


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    x = np.linspace(0, 1, 600)
    for ax, (s, n) in zip(axes, [(13, 20), (1300, 2000)]):
        print(f"\n{s} successes in {n} (data mean {s / n:.3f}):")
        for name, (a, b) in PRIORS.items():
            post = stats.beta(*beta_binomial_update(a, b, s, n))
            lo, hi = post.interval(0.95)
            print(f"  {name:24s} posterior mean={post.mean():.3f}  95% interval=({lo:.3f}, {hi:.3f})")
            ax.plot(x, post.pdf(x), label=name)
        ax.set_title(f"n = {n}")
        ax.set_xlabel(r"$\theta$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Posteriors under different priors: the prior matters less as n grows")
    fig.tight_layout()
    fig.savefig(OUT / "05_prior_sensitivity.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

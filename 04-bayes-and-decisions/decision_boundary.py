"""Bayes decision boundary for two 1-D Gaussian classes, with and without unequal costs.

Run:  python 04-bayes-and-decisions/decision_boundary.py
Saves output/04_decision_boundary.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).resolve().parents[1] / "output"

# class 1 = healthy, class 2 = disease
MU1, MU2, SIGMA = 0.0, 2.0, 1.0
PRIOR2 = 0.3  # P(disease)


def posterior_disease(x, prior2=PRIOR2):
    j1 = stats.norm.pdf(x, MU1, SIGMA) * (1 - prior2)
    j2 = stats.norm.pdf(x, MU2, SIGMA) * prior2
    return j2 / (j1 + j2)


def boundary(threshold):
    """x where P(disease | x) = threshold."""
    return optimize.brentq(lambda x: posterior_disease(x) - threshold, -10, 10)


def threshold_from_costs(c_fp, c_fn):
    """Decide disease when P(disease|x) > c_fp / (c_fp + c_fn)."""
    return c_fp / (c_fp + c_fn)


def error_rates(x_star, prior2=PRIOR2):
    """Decide disease when x > x_star. Returns (false positive, false negative, total error)."""
    fp = stats.norm.sf(x_star, MU1, SIGMA)
    fn = stats.norm.cdf(x_star, MU2, SIGMA)
    return fp, fn, fp * (1 - prior2) + fn * prior2


def main():
    for label, (c_fp, c_fn) in {"equal costs": (1, 1), "false alarm 3x a miss": (3, 1)}.items():
        t = threshold_from_costs(c_fp, c_fn)
        xs = boundary(t)
        fp, fn, err = error_rates(xs)
        print(f"{label:22s} threshold={t:.2f} boundary x*={xs:.3f}  FP={fp:.3f} FN={fn:.3f} total error={err:.3f}")

    # The 0.5 boundary should minimise the total error probability.
    grid = np.linspace(-2, 4, 2001)
    best = grid[np.argmin([error_rates(x)[2] for x in grid])]
    print(f"grid minimiser of p(mistake): {best:.3f}   (boundary at 0.5: {boundary(0.5):.3f})")

    OUT.mkdir(exist_ok=True)
    x = np.linspace(-4, 6, 500)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, stats.norm.pdf(x, MU1, SIGMA) * (1 - PRIOR2), label="p(x, healthy)")
    ax.plot(x, stats.norm.pdf(x, MU2, SIGMA) * PRIOR2, label="p(x, disease)")
    ax.axvline(boundary(0.5), color="k", ls="--", label="boundary at 0.5")
    ax.axvline(boundary(0.75), color="C3", ls=":", label="boundary at 0.75")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(OUT / "04_decision_boundary.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

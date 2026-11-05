"""Posterior mean / median / mode as the minimisers of squared / absolute / zero-one loss.

Run:  python 04-bayes-and-decisions/posterior_summaries.py
"""
import numpy as np
from scipy import stats

POSTERIOR = stats.beta(4.0, 9.0)  # a skewed posterior so the three summaries differ


def best_action_by_loss(posterior=POSTERIOR, grid_size=4001):
    """Minimise the expected loss over a grid of candidate actions a."""
    theta = np.linspace(1e-4, 1 - 1e-4, grid_size)
    w = posterior.pdf(theta)
    w /= w.sum()  # discrete weights over the grid
    actions = theta
    sq = [(w * (theta - a) ** 2).sum() for a in actions]
    ab = [(w * np.abs(theta - a)).sum() for a in actions]
    # zero-one loss on a grid: pay 1 unless within half a grid cell of the truth,
    # i.e. expected loss = 1 - posterior mass near a.
    half = (theta[1] - theta[0]) / 2
    zo = [1.0 - w[np.abs(theta - a) <= half].sum() for a in actions]
    return actions[np.argmin(sq)], actions[np.argmin(ab)], actions[np.argmin(zo)]


def main():
    mean, median, mode = best_action_by_loss()
    a, b = POSTERIOR.args
    print(f"squared error  -> {mean:.4f}   posterior mean   = {POSTERIOR.mean():.4f}")
    print(f"absolute error -> {median:.4f}   posterior median = {POSTERIOR.median():.4f}")
    print(f"zero-one       -> {mode:.4f}   posterior mode   = {(a - 1) / (a + b - 2):.4f}")


if __name__ == "__main__":
    main()

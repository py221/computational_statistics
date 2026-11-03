"""MoM, MLE and MAP for a Bernoulli parameter: closed forms vs numerical optimisation.

Run:  python 03-estimation/bernoulli_mle_map.py
Saves output/03_bernoulli_likelihood.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

OUT = Path(__file__).resolve().parents[1] / "output"

FLIPS = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0])  # 13 heads in 20


def log_likelihood(theta, x):
    s, n = x.sum(), x.size
    return s * np.log(theta) + (n - s) * np.log1p(-theta)


def log_prior_beta(theta, a, b):
    return (a - 1) * np.log(theta) + (b - 1) * np.log1p(-theta)  # up to a constant


def mom_closed_form(x):
    return x.mean()


def mle_closed_form(x):
    return x.mean()


def map_closed_form(x, a, b):
    return (x.sum() + a - 1) / (x.size + a + b - 2)


def mle_numeric(x):
    res = optimize.minimize_scalar(lambda t: -log_likelihood(t, x), bounds=(1e-9, 1 - 1e-9), method="bounded",
                                   options={"xatol": 1e-10})
    return res.x


def map_numeric(x, a, b):
    res = optimize.minimize_scalar(lambda t: -(log_likelihood(t, x) + log_prior_beta(t, a, b)),
                                   bounds=(1e-9, 1 - 1e-9), method="bounded", options={"xatol": 1e-10})
    return res.x


def main():
    x = FLIPS
    print(f"n={x.size}, heads={x.sum()}")
    print(f"MoM  closed form : {mom_closed_form(x):.4f}")
    print(f"MLE  closed form : {mle_closed_form(x):.4f}   numeric: {mle_numeric(x):.4f}")
    for a, b in [(1, 1), (2, 2), (10, 10), (8, 2)]:
        print(f"MAP  Beta({a:>2},{b:>2})  closed form: {map_closed_form(x, a, b):.4f}   numeric: {map_numeric(x, a, b):.4f}")

    OUT.mkdir(exist_ok=True)
    t = np.linspace(0.01, 0.99, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ll = log_likelihood(t, x)
    ax.plot(t, np.exp(ll - ll.max()), label="likelihood (scaled)")
    for a, b in [(2, 2), (10, 10)]:
        lp = ll + log_prior_beta(t, a, b)
        ax.plot(t, np.exp(lp - lp.max()), label=f"likelihood x Beta({a},{b}) prior")
    ax.axvline(x.mean(), color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\theta$")
    ax.set_title("MLE is the likelihood peak; MAP shifts it towards the prior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "03_bernoulli_likelihood.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

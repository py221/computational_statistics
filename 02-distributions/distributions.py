"""pmf / pdf / cdf of common distributions and a numerical moment check.

Run:  python 02-distributions/distributions.py
Saves a figure to output/02_distributions.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

OUT = Path(__file__).resolve().parents[1] / "output"


def numeric_moments(dist, lower=None, upper=None):
    """Mean, variance, skewness, kurtosis by integrating the pdf.

    Skewness and kurtosis follow the definitions in notes.md:
    alpha_3 = E[(X-mu)^3] / sigma^3 and alpha_4 = E[(X-mu)^4] / sigma^4.
    """
    lo = dist.ppf(1e-10) if lower is None else lower
    hi = dist.ppf(1 - 1e-10) if upper is None else upper

    def expect(g):
        return integrate.quad(lambda x: g(x) * dist.pdf(x), lo, hi, limit=200)[0]

    mean = expect(lambda x: x)
    var = expect(lambda x: (x - mean) ** 2)
    skew = expect(lambda x: (x - mean) ** 3) / var**1.5
    kurt = expect(lambda x: (x - mean) ** 4) / var**2
    return mean, var, skew, kurt


def cdf_vs_integral(dist, x):
    """P(X <= x) from the library cdf and by integrating the pdf."""
    lo = dist.ppf(1e-12)
    return dist.cdf(x), integrate.quad(dist.pdf, lo, x)[0]


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))

    # Discrete families: pmf as stems, cdf as steps.
    discrete = {
        "Bernoulli(0.3)": (stats.bernoulli(0.3), np.arange(0, 2)),
        "Binomial(10, 0.3)": (stats.binom(10, 0.3), np.arange(0, 11)),
        "Poisson(3)": (stats.poisson(3), np.arange(0, 13)),
    }
    for col, (name, (d, ks)) in enumerate(discrete.items()):
        axes[0, col].vlines(ks, 0, d.pmf(ks), colors="C0", lw=4)
        axes[0, col].set_title(f"{name}: pmf")
        axes[1, col].step(ks, d.cdf(ks), where="post", color="C0")
        axes[1, col].set_title(f"{name}: cdf")

    # Continuous families on one pair of axes.
    cont = {
        "Normal(0, 1)": stats.norm(0, 1),
        "Exponential(1)": stats.expon(scale=1.0),
        "Beta(2, 5)": stats.beta(2, 5),
    }
    xs = np.linspace(-4, 8, 600)
    for (name, d), c in zip(cont.items(), ["C0", "C1", "C2"]):
        axes[0, 3].plot(xs, d.pdf(xs), color=c, label=name)
        axes[1, 3].plot(xs, d.cdf(xs), color=c, label=name)
    axes[0, 3].set_title("continuous: pdf")
    axes[1, 3].set_title("continuous: cdf")
    axes[0, 3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "02_distributions.png", dpi=110)
    plt.close(fig)

    print("cdf = integral of pdf:")
    for name, d in cont.items():
        a, b = cdf_vs_integral(d, 0.8)
        print(f"  {name:16s} F(0.8) library={a:.6f}  integrated={b:.6f}")

    print("moments (numerical integration vs closed form):")
    closed = {
        "Normal(0, 1)": (0.0, 1.0, 0.0, 3.0),
        "Exponential(1)": (1.0, 1.0, 2.0, 9.0),
    }
    for name in closed:
        got = numeric_moments(cont[name], lower=-12 if "Normal" in name else 0, upper=40)
        print(f"  {name:16s} numeric={np.round(got, 4)}  closed={closed[name]}")


if __name__ == "__main__":
    main()

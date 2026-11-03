"""Method of moments vs maximum likelihood for a Gamma(shape, scale) sample.

Run:  python 03-estimation/mom_vs_mle.py
"""
import numpy as np
from scipy import stats


def gamma_mom(x):
    """Match mean = k*s and variance = k*s^2."""
    m, v = x.mean(), x.var()
    return m**2 / v, v / m  # shape k, scale s


def gamma_mle(x):
    """Numerical MLE with the location fixed at 0."""
    k, _, s = stats.gamma.fit(x, floc=0)
    return k, s


def compare(true_k=3.0, true_s=2.0, n=100, reps=500, seed=1):
    rng = np.random.default_rng(seed)
    mom = np.empty((reps, 2))
    mle = np.empty((reps, 2))
    for i in range(reps):
        x = rng.gamma(true_k, true_s, size=n)
        mom[i] = gamma_mom(x)
        mle[i] = gamma_mle(x)
    truth = np.array([true_k, true_s])
    return {
        "MoM": {"mean": mom.mean(0), "mse": ((mom - truth) ** 2).mean(0)},
        "MLE": {"mean": mle.mean(0), "mse": ((mle - truth) ** 2).mean(0)},
        "truth": truth,
    }


def main():
    res = compare()
    print("truth (shape, scale):", res["truth"])
    for name in ("MoM", "MLE"):
        print(f"{name}: mean estimate={np.round(res[name]['mean'], 3)}  MSE={np.round(res[name]['mse'], 4)}")


if __name__ == "__main__":
    main()

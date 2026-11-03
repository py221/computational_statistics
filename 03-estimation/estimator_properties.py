"""Simulating bias, variance, MSE and consistency of estimators.

Run:  python 03-estimation/estimator_properties.py
"""
import numpy as np


def variance_estimators(n=10, sigma2=4.0, reps=200_000, seed=0):
    """Compare dividing the sum of squares by n-1, n and n+1 for normal data.

    Returns {divisor_name: (bias, variance, mse)}.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, np.sqrt(sigma2), size=(reps, n))
    ss = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    out = {}
    for name, d in [("n-1 (unbiased)", n - 1), ("n (MLE)", n), ("n+1 (min MSE)", n + 1)]:
        est = ss / d
        bias = est.mean() - sigma2
        var = est.var()
        mse = ((est - sigma2) ** 2).mean()
        out[name] = (bias, var, mse)
    return out


def consistency(sizes=(10, 100, 1000, 10000), eps=0.1, reps=2000, theta=0.3, seed=0):
    """P(|mean - theta| > eps) for the Bernoulli sample mean as n grows."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        means = rng.binomial(n, theta, size=reps) / n
        out[n] = float(np.mean(np.abs(means - theta) > eps))
    return out


def main():
    print("Variance estimators (n=10, true sigma^2=4):")
    for name, (b, v, m) in variance_estimators().items():
        print(f"  divisor {name:15s} bias={b:+.4f}  var={v:.4f}  MSE={m:.4f}  (check: var+bias^2={v + b * b:.4f})")
    print("\nConsistency of the Bernoulli sample mean, P(|mean - theta| > 0.1):")
    for n, p in consistency().items():
        print(f"  n={n:>6d}  {p:.4f}")


if __name__ == "__main__":
    main()

"""Bayes' rule: a diagnostic-test posterior and a grid posterior for a coin.

Run:  python 04-bayes-and-decisions/bayes_rule.py
"""
import numpy as np


def posterior_disease(prevalence, sensitivity, false_positive_rate):
    """P(D | +) = P(+|D)P(D) / P(+)."""
    evidence = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
    return sensitivity * prevalence / evidence


def grid_posterior(heads, n, prior_pdf, grid_size=2001):
    """posterior(theta) proportional to likelihood(theta) * prior(theta), normalised on a grid."""
    theta = np.linspace(0.0, 1.0, grid_size)
    likelihood = theta**heads * (1 - theta) ** (n - heads)
    unnorm = likelihood * prior_pdf(theta)
    post = unnorm / np.trapezoid(unnorm, theta)
    return theta, post


def main():
    print("P(disease | +) =", round(posterior_disease(0.01, 0.95, 0.05), 4))
    # Repeating a positive test: yesterday's posterior is today's prior.
    p = 0.01
    for k in range(1, 4):
        p = posterior_disease(p, 0.95, 0.05)
        print(f"  after {k} independent positive test(s): {p:.4f}")

    theta, post = grid_posterior(13, 20, lambda t: np.ones_like(t))
    mean = np.trapezoid(theta * post, theta)
    print(f"\nCoin, 13 heads in 20, uniform prior: posterior mean={mean:.4f}, mode={theta[post.argmax()]:.4f}")


if __name__ == "__main__":
    main()

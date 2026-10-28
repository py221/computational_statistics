"""Probability rules on two fair dice, by exact enumeration and by simulation.

Run:  python 01-probability/dice_rules.py
"""
from fractions import Fraction
from itertools import product

import numpy as np

# Sample space: all ordered pairs of faces, each outcome equally likely.
SAMPLE_SPACE = list(product(range(1, 7), repeat=2))


def prob(event):
    """P(event) where `event` is a predicate on an outcome (d1, d2)."""
    hits = sum(1 for w in SAMPLE_SPACE if event(w))
    return Fraction(hits, len(SAMPLE_SPACE))


def cond_prob(a, b):
    """P(A | B) = P(A and B) / P(B)."""
    return prob(lambda w: a(w) and b(w)) / prob(b)


def sum_is_7(w):
    return w[0] + w[1] == 7


def first_is_even(w):
    return w[0] % 2 == 0


def sum_is_8(w):
    return w[0] + w[1] == 8


def first_is_6(w):
    return w[0] == 6


def marginal_of_first_die():
    """Marginal of die 1 from the joint table: sum out die 2."""
    return {x: sum(Fraction(1, 36) for y in range(1, 7)) for x in range(1, 7)}


def simulate(n=200_000, seed=0):
    """Estimate P(sum = 7) and P(sum = 7 | first die even) by simulation."""
    rng = np.random.default_rng(seed)
    d = rng.integers(1, 7, size=(n, 2))
    total = d.sum(axis=1)
    even = d[:, 0] % 2 == 0
    return {
        "P(sum=7)": float(np.mean(total == 7)),
        "P(sum=7 | first even)": float(np.mean(total[even] == 7)),
    }


def main():
    print("Sample space size:", len(SAMPLE_SPACE))
    print("P(sum = 7)               =", prob(sum_is_7))
    print("P(first die even)        =", prob(first_is_even))

    # Independence: sum=7 and first die even are independent for fair dice.
    joint = prob(lambda w: sum_is_7(w) and first_is_even(w))
    print("P(sum=7, first even)     =", joint,
          "  product of marginals =", prob(sum_is_7) * prob(first_is_even),
          "-> independent" if joint == prob(sum_is_7) * prob(first_is_even) else "-> dependent")

    # Sum=8 and first=6 are neither independent nor disjoint.
    joint = prob(lambda w: sum_is_8(w) and first_is_6(w))
    print("P(sum=8, first=6)        =", joint,
          "  product of marginals =", prob(sum_is_8) * prob(first_is_6))

    # Multiplication rule: P(A, B) = P(A) P(B | A)
    lhs = prob(lambda w: sum_is_8(w) and first_is_6(w))
    rhs = prob(first_is_6) * cond_prob(sum_is_8, first_is_6)
    print("multiplication rule      :", lhs, "==", rhs, lhs == rhs)

    # Inclusion-exclusion
    union = prob(lambda w: sum_is_7(w) or first_is_even(w))
    ie = prob(sum_is_7) + prob(first_is_even) - prob(lambda w: sum_is_7(w) and first_is_even(w))
    print("inclusion-exclusion      :", union, "==", ie, union == ie)

    print("Marginal of die 1        :", {k: str(v) for k, v in marginal_of_first_die().items()})
    print("Simulation               :", simulate())


if __name__ == "__main__":
    main()

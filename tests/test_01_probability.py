from fractions import Fraction

import dice_rules as d


def test_known_probabilities():
    assert d.prob(d.sum_is_7) == Fraction(1, 6)
    assert d.prob(d.first_is_even) == Fraction(1, 2)


def test_independence_and_dependence():
    p_joint = d.prob(lambda w: d.sum_is_7(w) and d.first_is_even(w))
    assert p_joint == d.prob(d.sum_is_7) * d.prob(d.first_is_even)
    p_joint = d.prob(lambda w: d.sum_is_8(w) and d.first_is_6(w))
    assert p_joint != d.prob(d.sum_is_8) * d.prob(d.first_is_6)


def test_multiplication_rule_and_inclusion_exclusion():
    lhs = d.prob(lambda w: d.sum_is_8(w) and d.first_is_6(w))
    assert lhs == d.prob(d.first_is_6) * d.cond_prob(d.sum_is_8, d.first_is_6)
    union = d.prob(lambda w: d.sum_is_7(w) or d.first_is_even(w))
    inter = d.prob(lambda w: d.sum_is_7(w) and d.first_is_even(w))
    assert union == d.prob(d.sum_is_7) + d.prob(d.first_is_even) - inter


def test_marginal_sums_to_one():
    assert sum(d.marginal_of_first_die().values()) == 1


def test_simulation_close_to_exact():
    sim = d.simulate(n=100_000, seed=3)
    assert abs(sim["P(sum=7)"] - 1 / 6) < 0.01

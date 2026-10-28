# 01 - Probability

Beliefs about uncertain events are expressed with probabilities. This stage sets up the vocabulary
(sample spaces, events, random variables) and the rules for combining probabilities that the rest of the path relies on.

## Sample space, outcomes and events

- **Outcome**: one possible result of a random experiment. Outcomes are mutually exclusive: exactly one occurs.
- **Sample space** $S$ (or $\Omega$): the set of all outcomes. It must be collectively exhaustive (nothing else can happen) and its elements mutually exclusive.
- **Event**: a subset of the sample space, that is, a set containing zero or more outcomes. We define the events we care about.

Example: flip a coin twice.

- Sample space: $S = \{HH, HT, TH, TT\}$
- Outcomes: each of $HH$, $HT$, $TH$, $TT$ (four, with probability $\tfrac14$ each for fair coins)
- Event "at least one head": $A = \{HH, HT, TH\}$, so $P(A) = \tfrac34$
- Event "both flips match": $E = \{HH, TT\}$, so $P(E) = \tfrac12$

## Probability space and the axioms

A probability space combines three things: a sample space, an event space (a collection of events closed under complement and countable union), and a probability function $P$.

1. $0 \le P(E) \le 1$ for every event $E$.
2. $P(S) = 1$.
3. Countable additivity: for pairwise disjoint events $E_1, E_2, \dots$

$$P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)$$

Here $\bigcup$ means the union of the events $E_i$.

Useful consequences: $P(\bar{E}) = 1 - P(E)$ and, for any two events,

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

## Conditional probability, independence, disjointness

- **Conditional**: $P(A \mid B) = P(A, B) / P(B)$ for $P(B) > 0$.
- **Independent**: $P(A, B) = P(A)\,P(B)$, equivalently $P(A \mid B) = P(A)$.
- **Disjoint (mutually exclusive)**: $P(A, B) = 0$.

Here $P(A, B)$ is shorthand for $P(A \cap B)$.

Two warnings:

- Disjoint events with positive probability are **not** independent: knowing $A$ happened tells you $B$ did not.
- **Exchangeable** random variables have a joint distribution that does not change when you permute them: $P(X_1 = x_1, \dots, X_n = x_n) = P(X_{\pi(1)} = x_1, \dots, X_{\pi(n)} = x_n)$ for every permutation $\pi$. Independent and identically distributed (iid) variables are exchangeable, but exchangeable variables need not be independent (draws without replacement from an urn are the classic case).

## Multiplication rule

$$P(A \cap B) = P(A)\,P(B \mid A)$$

- If $A$ and $B$ are independent this reduces to $P(A \cap B) = P(A)\,P(B)$.
- If they are mutually exclusive, $P(A \cap B) = 0$ (the conditional factor is $0$).

## Factoring joint probabilities (chain rule)

The chain rule breaks a joint probability into pieces that are tractable or belong to known distributions:

$$P(A, B) = P(A \mid B)\,P(B)$$

$$P(A, B, C) = P(A \mid B, C)\,P(B \mid C)\,P(C)$$

Any ordering of the variables works, which is what makes Bayes' rule possible later (stage 04).

## Random variables

A random variable is a (usually real-valued) function that maps outcomes to a measurable space:

$$X : \Omega \to \mathbb{R}$$

Its distribution is induced by the probability of the outcomes that land in a set $S$:

$$P(X \in S) = P\left(\{\omega \in \Omega \mid X(\omega) \in S\}\right)$$

Example: for two coin flips let $X$ be the number of heads. Then $X(HH) = 2$, $X(HT) = X(TH) = 1$, $X(TT) = 0$, and $P(X = 1) = P(\{HT, TH\}) = \tfrac12$.

## Experiment, trial, random sample

- **Random experiment**: an activity with an observable, uncertain result.
- **Trial**: one execution of the experiment.
- **Random sample**: the results of repeated trials, ideally independent and identically distributed.

Example: the experiment is "throw two dice and record the faces". A trial is one throw. Repeating it 500 times gives a random sample of 500 pairs.

## Marginal distribution

Given a joint distribution of several variables, the marginal distribution of a subset of them is obtained by summing (or integrating) out the others. For discrete $X$, $Y$:

$$P_X(x) = \sum_{y} P_{XY}(x, y)$$

The marginal of $X$ "ignores" $Y$ by adding up its effect over every value $y$. Marginals are covered further in stage 02.

## Scripts

- `dice_rules.py` - computes the rules above exactly for two dice (enumeration) and checks them with simulation. Run `python 01-probability/dice_rules.py` from the repo root.

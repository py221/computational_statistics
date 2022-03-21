# Intro to Bayesian Statistics

## Belif and Probability

### Sample space, outcomes, and events

1. outcomes
    - mutually exclusive and exhaustie list of possible results ina modlel
2. events
    - setscontaing zero or more outcome
    - defined as we are interested in
3. sample space
    - set of all possible outcome
    - must be collectively exhaustive and mutally exclusive

-   e.g.

> outcomes - {(H,T), (H, H), (H, H)}
> Events - {(H,T: 1). (H, H: 2)}
> Sample space - {(H, H), (T , T), (H, T), (T, H)}

### Probability Sapce & Probability

```
combinaton of a sample space, event space, and probability funcion
```

-   0 &le; P(E) &le; 1
    <br>
-   P(S) = 1 where S is the sample space
    <br>
-   $P(\bigcup_{i=1}^\infty E_i) = \sum_{i=1}^\infty P(E_i)$
    > &bigcup; means union of X<sub>i</sub>

## Probability Rules

### Multiplication Rule

1.  Dependent
    -   P(A &cap; B) = P(A) \* P(B|A)
2.  Independent
    -   P(A &cap; B) = P(A) \* P(B)
3.  Mutually exclusive
    -   P(A &cap; B) = 0

### Conditional, Independent, disjoint,exchangeable

1. Conditional
    - $$P(A|B) = \frac{P(A,B)}{P(B)}$$
2. Independent
    - $$P(A,B) = P(A) \ast P(B)$$
3. Disjoint
    - $$P(A,B) = 0$$
4. Exhangeable
    - $$P(A then B) = P(B then A)$$

### Factoring joint probabilitie

-   chain rule for simplifying joint probabilities to tractable/know distributions

$$P(A,B) = P(A|B) \ast P(B)$$

$$P(A,B,C) = P(A|B,C) \ast P(B,C) = P(A|B,C) \ast P(B|C) \ast P(C)$$

### Random Varable

Random variables are (often) real-valued functions mapping outcomes to a measurable space. In set notation, this looks like:

$$X: \Omega \to \mathbb{R}$$

This mapping defines the probability giving by X in a measurable set as:

$$P(X \in S) = P({x \in \Omega | X(x) \in S})$$

### Random sample, experiment, trial

1. Random Sample
    - unbiased realization of outcomes
2. Random experiment
    - actity w observable result
3. Trial
    - one or more experiments

-   e.g.
    > Experiment = throw 2 dices 500 times
    > Trial = single/multiple trial(s) for experiment
    > Random sample = 500 realizations of two dices being thrown

### Marginal Distribution

-   probability distribution containing the subset
-   e.g.
    -   marginal distri for x

## Distributions

### Probability Distributions ... pdf pmf

#### Distributions & Random Variables

1. Radom Variable
    - def - real-valued funcs mapping events to measurable space
    - e.g. RV
        - Random Process funcc
            - $$P(X \in S) = P({x \in \Omega | X(x) \in S})$$
        - Event Space
            - $$\\(\mathcal{F} = \{S, \bar{S}, E, \bar{E} \})$$
            - $$\\(E = \{(HH),(TT)\})$$
        - Prob Func
            - \\(X \sim Bern(\theta)\\)
            - $$f(x;\theta) = \theta^x(1-\theta)^{(1-x)}$$
2. Discrete Random Var
    - characteriezed by countable outcome spaces
    - pmd...probability mass function
        1. $f_x(x) \ge 0$, for all $x \in \mathbb{R}_X$.
        2. $\sum_{x \in X} f_x(x) = 1$
        3. $F_X(b) - F_X(a) = \sum_{x=a}^b f(x), a < b, a, b \in \mathbb{R}$
    - cdf f discrete RV.... cumulative distribution function
        1. $F_X(b) = \sum_{x \in [-\infty,b]} f_X(x)$
        2. properties
            - non-decreasing: $F_X(x) \le F_X(y)$, for all $x \le y$
            - right-continuous: $\lim_{x \downarrow x_0^+} F_X(x) = F_X(x_0)$
            - positive, with range [0,1]
3. Conti Random Var
    - range = uncoutable subset of real space w prob in the range [0,\\(\infty\\)]
    - pdf...probability dense function
        1. $ \\(f_x(x) \ge 0\\), for all \\(x \in \mathbb{R}\_X\\) $
        3. \\(\int\_{-\infty}^\infty f_x(x)dx = 1\\)
        4. \\(F*X(b) - F_X(a) = \int*{a}^b f(x)dx,\ a < b, (a, b) \in \mathbb{R}\\)
    - cdf f conti RV.... cumulative distribution func
        1. \\(P(X \le x) = F*X(x) = \int*{-\infty}^x f(u)du\\)
            - reverting back to pdf
                - \\(f(x) = \frac{dF(x)}{dx}\\)
    - e.g. Normal distri
        - $$X \sim Norm(\mu,\sigma^2) \text{, where}$$
        - $$f_x(x,\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, (x,\mu,\sigma^2) \in (\mathbb{R},\mathbb{R},\mathbb{R}_+)$$

### moments, expectations and variances

```
> Moments can be used to estimate the parameters defining a distribution
e.g. Expectation, Variance, Skewness, Kurtosis
```

-   Expectation (Expeted Value)
    -   average value of the func under a probability distribution
    -   discrete
        -   $$E[f] = \sum_x f(x)^r p(x)$$
    -   conti
        -   $$E[f] = \int f(x)^r p(x) dx.$$
-   Variance (second moments... [X-E[X]^2])
    -   $$Var(X) = E[(X-E[X])^2] = \int (X-E[X])^2 p(x) dx\text{ in the continuous case}$$
-   Skewness... measuer symmetry
    -   $$\alpha_3 = \frac{E(X-E[X])^3}{(\sqrt{Var(X)})^3}$$
-   Kurtosis... measure "peakiness"
    -   $$\alpha_4 = \frac{E(X-E[X])^4}{(\sqrt{Var(X)})^4}$$

> Properties
    >> \\(Var(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2\\)  
    >> \\(Var(c) = 0\\), where c is a constant
    >> \\(Var(aX_1 + bX_2) = a^2Var(X_1) + b^2Var(X_2)\\)

### jjoint distriutions, expectations and covariances w 2 Variables

```
random variable"s" w different distriution
```

- $$P_{XY}(x,y) = P(X=x,Y=y)$$
- $$E[f(x,y)] = \int_{x,y} f(x,y) p(x,y) dxdy$$

### Marginal / Conditional / Covar

#### Marginal distribution

```
given a joint distribution of X, Y
```

- formula (marginal of x... bc summimg up effectof y)
    - \\(P*X(x) = \sum*{all y*i} P*{XY}(x,y_i)\\)

#### Conditional distribution

$$P_{XY}(x_i|y_j) = \frac{P_{XY}(x_i,y_j)}{P_Y(y_i)}$$

The conditional expectation is given by

$$E[X|Y=y_j] = \sum_{X} x_iP_{X|Y}(x_i|y_j)$$

or in the continuous case for a function g:

$$E[g(Y)|x] = \int_{-\infty}^{\infty} g(y)f(y|x)dy$$

#### Independence

It can be shown that if X and Y are independent, there exists some functions g(x) and h(y) such that:

$$f(x,y) = g(x)h(y)\text{ for all (x,y)}$$

In the discrete case, if we can find a pair (x,y) that violate the product rule, the random variables are dependent.

#### Covariance

$$cov[x,y] = E[(x-E[x])(y-E[y])] = E_{x,y}[xy]-E_x[x]E_y[y]$$

#### Correlation

$$\rho_{XY} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}$$

## MoM & MLE (Estimation)

# bayesian infernce w mcmc (Markov Chain Monte Carlo)

# pymc3 for bayesian modeling and inferences

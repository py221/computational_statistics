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
-   <img src="https://render.githubusercontent.com/render/math?math=P(\bigcup_{i=1}^\infty E_i) = \sum_{i=1}^\infty P(E_i)">
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
    - <img src="https://render.githubusercontent.com/render/math?math=P(A|B)=\frac{P(A,B)}{P(B)}">
2. Independent
    - <img src="https://render.githubusercontent.com/render/math?math=P(A,B) = P(A) \ast P(B)">
3. Disjoint
    - <img src="https://render.githubusercontent.com/render/math?math=P(A,B) = 0">
4. Exhangeable
    - <img src="https://render.githubusercontent.com/render/math?math=P(A then B) = P(B then A)">

### Factoring joint probabilitie

-   chain rule for simplifying joint probabilities to tractable/know distributions
<br>
<img src="https://render.githubusercontent.com/render/math?math=P(A,B) = P(A|B) \ast P(B)">
<br>
<img src="https://render.githubusercontent.com/render/math?math=P(A,B,C) = P(A|B,C) \ast P(B,C) = P(A|B,C) \ast P(B|C) \ast P(C)">

### Random Varable

Random variables are (often) real-valued functions mapping outcomes to a measurable space. In set notation, this looks like:

<img src="https://render.githubusercontent.com/render/math?math=X: \Omega \to \mathbb{R}">

This mapping defines the probability giving by X in a measurable set as:

<img src="https://render.githubusercontent.com/render/math?math=P(X \in S) = P({x \in \Omega | X(x) \in S})">

### Random sample, experiment, trial

1. Random Sample
    - unbiased realization of outcomes
2. Random experiment
    - actity w observable result
3. Trial
    - one or more experiments

-   e.g.
    > Experiment = throw 2 dices 500 times <br>
    > Trial = single/multiple trial(s) for experiment <br>
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
            - <img src="https://render.githubusercontent.com/render/math?math=P(X \in S) = P({x \in \Omega | X(x) \in S})">
        - Event Space
            - <img src="https://render.githubusercontent.com/render/math?math=\\(\mathcal{F} = \{S, \bar{S}, E, \bar{E} \})">
            - <img src="https://render.githubusercontent.com/render/math?math=\\(E = \{(HH),(TT)\})">
        - Prob Func
            - \\(X \sim Bern(\theta)\\)
            - <img src="https://render.githubusercontent.com/render/math?math=f(x;\theta) = \theta^x(1-\theta)^{(1-x)}">
2. Discrete Random Var
    - characteriezed by countable outcome spaces
    - pmd...probability mass function
        1. <img src="https://render.githubusercontent.com/render/math?math=f_x(x) \ge 0$, for all $x \in \mathbb{R}_X">
        2. <img src="https://render.githubusercontent.com/render/math?math=\sum_{x \in X} f_x(x) = 1">
        3. <img src="https://render.githubusercontent.com/render/math?math=F_X(b) - F_X(a) = \sum_{x=a}^b f(x), a < b, a, b \in \mathbb{R}">
    - cdf f discrete RV.... cumulative distribution function
        1. <img src="https://render.githubusercontent.com/render/math?math=F_X(b) = \sum_{x \in [-\infty,b]} f_X(x)">
        2. properties
            - non-decreasing: <img src="https://render.githubusercontent.com/render/math?math=F_X(x) \le F_X(y)$, for all $x \le y">
            - right-continuous: <img src="https://render.githubusercontent.com/render/math?math=\lim_{x \downarrow x_0^+} F_X(x) = F_X(x_0)">
            - positive, with range [0,1]
3. Conti Random Var
    - range = uncoutable subset of real space w prob in the range <img src="https://render.githubusercontent.com/render/math?math=[0,\\(\infty\\)]">
    - pdf...probability dense function
        1. <img src="https://render.githubusercontent.com/render/math?math=\\(f_x(x) \ge 0\\), for all \\(x \in \mathbb{R}\_X\\)">
        3. <img src="https://render.githubusercontent.com/render/math?math=\\(\int\_{-\infty}^\infty f_x(x)dx = 1\\)">
        4. <img src="https://render.githubusercontent.com/render/math?math=\\(F*X(b) - F_X(a) = \int*{a}^b f(x)dx,\ a < b, (a, b) \in \mathbb{R}\\)">
    - cdf f conti RV.... cumulative distribution func
        1. <img src="https://render.githubusercontent.com/render/math?math=\\(P(X \le x) = F*X(x) = \int*{-\infty}^x f(u)du\\)">
            - reverting back to pdf
                - <img src="https://render.githubusercontent.com/render/math?math=\\(f(x) = \frac{dF(x)}{dx}\\)">
    - e.g. Normal distri
        - <img src="https://render.githubusercontent.com/render/math?math=X \sim Norm(\mu,\sigma^2) \text{, where}">
        - <img src="https://render.githubusercontent.com/render/math?math=f_x(x,\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, (x,\mu,\sigma^2) \in (\mathbb{R},\mathbb{R},\mathbb{R}_+)">

### moments, expectations and variances

```
> Moments can be used to estimate the parameters defining a distribution
e.g. Expectation, Variance, Skewness, Kurtosis
```

-   Expectation (Expeted Value)
    -   average value of the func under a probability distribution
    -   discrete
        -   <img src="https://render.githubusercontent.com/render/math?math=E[f] = \sum_x f(x)^r p(x)">
    -   conti
        -   <img src="https://render.githubusercontent.com/render/math?math=E[f] = \int f(x)^r p(x) dx.">
-   Variance (second moments... [X-E[X]^2])
    -   <img src="https://render.githubusercontent.com/render/math?math=Var(X) = E[(X-E[X])^2] = \int (X-E[X])^2 p(x) dx\text{ in the continuous case}">
-   Skewness... measuer symmetry
    -   <img src="https://render.githubusercontent.com/render/math?math=\alpha_3 = \frac{E(X-E[X])^3}{(\sqrt{Var(X)})^3}">
-   Kurtosis... measure "peakiness"
    -   <img src="https://render.githubusercontent.com/render/math?math=\alpha_4 = \frac{E(X-E[X])^4}{(\sqrt{Var(X)})^4}">

> Properties
    >> <img src="https://render.githubusercontent.com/render/math?math=\\(Var(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2\\)">
    >> <img src="https://render.githubusercontent.com/render/math?math=\\(Var(c) = 0\\), where c is a constant">
    >> <img src="https://render.githubusercontent.com/render/math?math=\\(Var(aX_1 + bX_2) = a^2Var(X_1) + b^2Var(X_2)\\)">

### jjoint distriutions, expectations and covariances w 2 Variables

```
random variable"s" w different distriution
```

- <img src="https://render.githubusercontent.com/render/math?math=P_{XY}(x,y) = P(X=x,Y=y)">
- <img src="https://render.githubusercontent.com/render/math?math=E[f(x,y)] = \int_{x,y} f(x,y) p(x,y) dxdy">

### Marginal / Conditional / Covar

#### Marginal distribution

```
given a joint distribution of X, Y
```

- formula (marginal of x... bc summimg up effectof y)
    - <img src="https://render.githubusercontent.com/render/math?math=\\(P*X(x) = \sum*{all y*i} P*{XY}(x,y_i)\\)">

#### Conditional distribution

<img src="https://render.githubusercontent.com/render/math?math=P_{XY}(x_i|y_j) = \frac{P_{XY}(x_i,y_j)}{P_Y(y_i)}">

The conditional expectation is given by

<img src="https://render.githubusercontent.com/render/math?math=E[X|Y=y_j] = \sum_{X} x_iP_{X|Y}(x_i|y_j)">

or in the continuous case for a function g:

<img src="https://render.githubusercontent.com/render/math?math=E[g(Y)|x] = \int_{-\infty}^{\infty} g(y)f(y|x)dy">

#### Independence

It can be shown that if X and Y are independent, there exists some functions g(x) and h(y) such that:

<img src="https://render.githubusercontent.com/render/math?math=f(x,y) = g(x)h(y)\text{ for all (x,y)}">

In the discrete case, if we can find a pair (x,y) that violate the product rule, the random variables are dependent.

#### Covariance

<img src="https://render.githubusercontent.com/render/math?math=cov[x,y] = E[(x-E[x])(y-E[y])] = E_{x,y}[xy]-E_x[x]E_y[y]">

#### Correlation

<img src="https://render.githubusercontent.com/render/math?math=\rho_{XY} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}">

## MoM & MLE (Estimation)
### Estimators Properties 
1. Consistency (Variance)
    - <img src="https://render.githubusercontent.com/render/math?math=P(|\theta_n-\theta |>0) \to 0 \text{ as } n \to \\infty">
    - As sample gets to infinity, the estimator converges in probability to <img src="https://render.githubusercontent.com/render/math?math=\\(\theta\\)">
2. Bias
    - <img src="https://render.githubusercontent.com/render/math?math=\\(\theta\\)"> 
    is unbiased if 
    <img src="https://render.githubusercontent.com/render/math?math=E(\theta_n)=\theta">
3. Efficiency 
    - Estimators w. ...
        - lowest possible vaiance
        - unbiased
4. MSE (Mean Squaed Error) 
    - <img src="https://render.githubusercontent.com/render/math?math=MSE={Variance}"> + <img src="https://render.githubusercontent.com/render/math?math={bias}^2">
    - measure of trade off b/t ...
        - accuracy (spread)
        - precision (location)

### MoM (Method of Moments) 
``` amouting tomatching population moments to sample momens ```
1. Formula
    - <img src="https://render.githubusercontent.com/render/math?math=E[f] = \int f(x)^r p(x) dx \approx \frac{1}{N} \sum f(x)^r p(x)">
    <br>
    where <img src="https://render.githubusercontent.com/render/math?math=\\(f(x)=x\\) and \\(r=1\\)">, amounting to 
    <br>
    <img src="https://render.githubusercontent.com/render/math?math=\mu \approx \bar{x}">

2. Properties 
    - \# of moments required = # of param est

3. e.g. <img src="https://render.githubusercontent.com/render/math?math=\\(X_i \sim Bern(\theta)\\)">
    - pmf
        - <img src="https://render.githubusercontent.com/render/math?math=f(x|\theta) = \theta^x(1-\theta)^{(1-x)}">
    - scenario 
        - N=20
        - {1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,0,1,1,1,0} (13 heads)
    - Mean & Var esitmation 
        - ...for population 
            - <img src="https://render.githubusercontent.com/render/math?math=E[X] = \sum_x x f(x) = \sum_x x [\theta^x(1-\theta)^{(1-x)}] = 0\ast(1-\theta) + 1\ast\theta = \theta">
            - <img src="https://render.githubusercontent.com/render/math?math=Var(X) = \theta(1-\theta)">
        - matching sample momnt to popu moment 
            - <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta} = \frac{1}{N} \sum_N x_i = \frac{13}{20}">



### MLE (Maximum Likelihood Estimation)
##### Assumption
> data results from independet & identically distributed obs from a population <br>
    
##### Goal 
> find a /theta that max the likelihood of observing the data

##### Likelihood func
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(\theta|x_1 ... x_n) = \prod_{i=1}^n f(x_i | \theta)">

- solving the below func
    - <img src="https://render.githubusercontent.com/render/math?math=\\(\frac{d}{d\theta}\mathcal{L}(\theta|{x})=0\\)">
    - making sure it's max and not on the boundary 

- Conversion to Log Likelihood 

    - Rsn

        - ease of cimputation 

    - formula

        - <img src="https://render.githubusercontent.com/render/math?math=\\(ln \mathcal{L}(\theta|x_1 ... x_n) = (\sum_{i=1}^n x_i) ln \theta + (\sum_{i=1}^n (1-x_i)) ln (1-\theta)\\)">

    - differentiating wrt to &theta;

        - <img src="https://render.githubusercontent.com/render/math?math=\\((\sum_{i=1}^n x_i)\frac{1}{\hat{\theta}} -(\sum_{i=1}^n (1-x_i))\frac{1}{1-\hat{\theta}}\\) = 0">



### MAP (Maximum a posteriori probability estimate)
``` augmented MLE using prior (additional informaton)```
- Prior
    
    - beleive we have for the distribution

- formula 

    - <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_{MAP} = arg max_{\theta} \mathcal{L}(\theta|x_1 ... x_n) \ast \pi (\theta)">

    - &pi;(&theta;) is the prior / additional information


## Decisions, loss and priors

## Priors 












# bayesian infernce w mcmc (Markov Chain Monte Carlo)

# pymc3 for bayesian modeling and inferences

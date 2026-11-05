# 04 - Bayes' rule and decisions

Bayes' rule turns "how likely is the data given a hypothesis" into "how likely is the hypothesis given the data".
Combined with a loss function it tells us how to act on that belief.

## Bayes' rule

From the chain rule (stage 01) the joint probability can be factored two ways:

1. $P(A, B) = P(A \mid B)\,P(B)$
2. $P(A, B) = P(B \mid A)\,P(A)$

Equating them and dividing by $P(B)$ gives

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

Any mix of discrete and continuous random variables can be handled the same way. With $X$ discrete and $Y$ continuous:

$$P(X = x \mid Y = y) = \frac{f_{Y \mid X = x}(y)\,P(X = x)}{f_Y(y)}$$

## Posterior, likelihood, prior, marginal

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{marginal}}$$

For $P(A \mid B) = \dfrac{P(B \mid A)\,P(A)}{P(B)}$:

1. **Posterior** is $P(A \mid B)$: belief about $A$ after seeing $B$.
2. **Likelihood** is $P(B \mid A)$: how probable the evidence is if $A$ is true.
3. **Prior** is $P(A)$: belief about $A$ before seeing $B$.
4. **Marginal** (evidence) is $P(B) = \sum_a P(B \mid a)P(a)$: a normalising constant.

The marginal does not depend on $A$, so for finding the shape of the posterior it can be dropped:

$$P(A \mid B) \;\propto\; P(B \mid A)\,P(A)$$

This "posterior is proportional to likelihood times prior" form is what grid approximations (this stage), conjugate updates (stage 05) and MCMC (stage 06) all exploit.

### Worked example: a diagnostic test

Suppose a disease has prevalence $P(D) = 0.01$, the test is positive for $95\%$ of the sick, $P(+ \mid D) = 0.95$, and for $5\%$ of the healthy, $P(+ \mid \bar{D}) = 0.05$.

$$P(D \mid +) = \frac{0.95 \cdot 0.01}{0.95 \cdot 0.01 + 0.05 \cdot 0.99} \approx 0.161$$

A positive test raises the probability from $1\%$ to about $16\%$, not to $95\%$: the prior matters. `bayes_rule.py` computes this and a grid posterior for a coin.

## Inference and decisions

Inference gives a posterior over classes (or parameters). A **decision** then maps each observation $x$ to a class.

### Minimising mistakes

For classes $C_1, \dots, C_k$ the goal for a binary decision is to minimise the probability of a mistake. If the input space is split into regions $R_1$ (decide $C_1$) and $R_2$ (decide $C_2$), then

$$p(\text{mistake}) = \int_{R_1} p(x, C_2)\,dx + \int_{R_2} p(x, C_1)\,dx$$

This is smallest when each $x$ is assigned to the class with the larger joint probability $p(x, C_k)$, equivalently the larger posterior $p(C_k \mid x)$.

### Rejection regions and decision boundaries

We choose a region for inclusion in one class or another. For example, we might decide that the optimal boundary in a disease classification is at

$$p(\text{disease} \mid \text{positive test}) > 0.75$$

The decision boundary must partition the input space: every $x$ lands in exactly one region.

### Loss functions

Misclassifications rarely cost the same. Let $L_{kj}$ be the loss for deciding class $j$ when the truth is class $k$. The expected loss is

$$E[L] = \sum_k \sum_j \int_{R_j} L_{kj}\,p(x, C_k)\,dx$$

and the optimal rule assigns $x$ to the class $j$ that minimises $\sum_k L_{kj}\,p(C_k \mid x)$.

For two classes, with cost $c_{FP}$ for a false alarm and $c_{FN}$ for a miss, we decide "disease" when

$$p(D \mid x)\,c_{FN} > \left(1 - p(D \mid x)\right)c_{FP} \iff p(D \mid x) > \frac{c_{FP}}{c_{FP} + c_{FN}}$$

So a threshold of $0.75$ corresponds to a false alarm being three times as costly as a miss. With equal costs the threshold is $0.5$. `decision_boundary.py` finds the boundary for two Gaussian classes and the resulting error.

### Loss functions for categorical outputs

These are the training losses used when a model outputs class probabilities $\hat{p}_k$ and the truth is class $y$:

1. **Cross-entropy (log) loss**, binary: $-\left[y\ln\hat{p} + (1-y)\ln(1-\hat{p})\right]$
2. **Categorical cross-entropy**, one-hot $y_k$: $-\sum_k y_k \ln \hat{p}_k$
3. **Focal loss**, which down-weights easy examples: $-(1 - \hat{p}_y)^{\gamma}\ln\hat{p}_y$ with focusing parameter $\gamma \ge 0$ (it reduces to cross-entropy at $\gamma = 0$)
4. **Hinge loss**, for margin classifiers with $y \in \{-1, +1\}$ and score $s$: $\max(0,\, 1 - y\,s)$

### Loss functions for continuous quantities (point estimates)

Choosing a single number $a$ to report for an unknown $\theta$ with posterior $p(\theta \mid x)$ is also a decision problem. The loss decides which summary of the posterior is optimal:

- squared error $(\theta - a)^2$ gives the posterior **mean**
- absolute error $|\theta - a|$ gives the posterior **median**
- zero-one loss (right or wrong) gives the posterior **mode**, which is the MAP estimate from stage 03

`posterior_summaries.py` verifies this by minimising each expected loss on a grid.

## Scripts

- `bayes_rule.py` - diagnostic-test posterior and a grid posterior for a coin ($\text{posterior} \propto \text{likelihood} \times \text{prior}$).
- `decision_boundary.py` - Bayes decision boundary for two Gaussian classes, error rate, and the effect of unequal costs.
- `posterior_summaries.py` - mean, median and mode as minimisers of squared, absolute and zero-one loss.

# 05 - Priors

The prior $\pi(\theta)$ encodes what we believe about a parameter before seeing data. Bayes' rule (stage 04) then gives

$$\pi(\theta \mid x) \;\propto\; \mathcal{L}(\theta \mid x)\,\pi(\theta)$$

This stage covers priors that keep the algebra closed-form (conjugate priors), how strongly a prior should speak (informative versus weak), and how to check whether conclusions depend on that choice (sensitivity).

## Conjugate priors

A prior is **conjugate** to a likelihood if the posterior belongs to the same family as the prior. Updating then reduces to adding numbers to the prior's parameters, with no integration.

### Beta-Binomial

Likelihood: $s$ successes in $n$ Bernoulli/Binomial trials with success probability $\theta$,

$$\mathcal{L}(\theta \mid s, n) \propto \theta^{s}(1 - \theta)^{n - s}$$

Prior: $\theta \sim \text{Beta}(a, b)$, with $\pi(\theta) \propto \theta^{a-1}(1-\theta)^{b-1}$. Multiplying,

$$\pi(\theta \mid s, n) \propto \theta^{a + s - 1}(1 - \theta)^{b + n - s - 1} \quad\Longrightarrow\quad \theta \mid s, n \sim \text{Beta}(a + s,\; b + n - s)$$

Reading $a$ and $b$ as "pseudo-counts" of prior successes and failures makes the update intuitive. Useful summaries of $\text{Beta}(\alpha, \beta)$:

$$\text{mean} = \frac{\alpha}{\alpha + \beta}, \qquad \text{mode} = \frac{\alpha - 1}{\alpha + \beta - 2}, \qquad \text{Var} = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$$

The posterior mean is a weighted average of the prior mean and the data mean $s/n$:

$$E[\theta \mid s, n] = \frac{a + s}{a + b + n} = \frac{a + b}{a + b + n}\cdot\frac{a}{a + b} + \frac{n}{a + b + n}\cdot\frac{s}{n}$$

so the prior's weight is $a + b$ "observations" and shrinks as $n$ grows. The mode formula is the MAP estimate from stage 03.

### Normal-Normal (known variance)

Data $x_1, \dots, x_n \sim \text{Norm}(\mu, \sigma^2)$ with $\sigma^2$ known. Prior $\mu \sim \text{Norm}(\mu_0, \tau_0^2)$. Working in precisions ($1/\text{variance}$), the posterior is normal with

$$\frac{1}{\tau_n^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}, \qquad \mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{n\,\bar{x}}{\sigma^2}\right)$$

Precisions add, and the posterior mean is a precision-weighted average of the prior mean and the sample mean $\bar{x}$.

### Others worth knowing

- Gamma-Poisson: a $\text{Gamma}(\alpha, \beta)$ prior on a Poisson rate gives a $\text{Gamma}(\alpha + \sum x_i,\; \beta + n)$ posterior.
- Dirichlet-Multinomial: the multi-category generalisation of Beta-Binomial.

`conjugate.py` implements the first two and checks them against a brute-force grid posterior.

## Informative versus weak priors

- **Informative prior**: carries real knowledge, such as earlier studies or physical limits. $\text{Beta}(40, 60)$ says "about $40\%$, worth roughly $100$ observations".
- **Weakly informative prior**: only rules out the absurd, with little influence on the posterior. $\text{Beta}(2, 2)$ gently favours the middle.
- **Flat / non-informative prior**: tries to add nothing. $\text{Beta}(1, 1)$ is uniform. Jeffreys' prior for a Bernoulli probability is $\text{Beta}(\tfrac12, \tfrac12)$, which is invariant to reparameterisation. "Flat" is not truly neutral: a uniform prior on $\theta$ is not uniform on, say, the log-odds.

A handy gauge of prior strength is its **effective sample size**: $a + b$ for a Beta prior, $\sigma^2 / \tau_0^2$ for a normal prior on a mean.

## Sensitivity analysis

Refit with several defensible priors and compare the posteriors.

- If conclusions barely move, the data dominate and the prior choice is not a concern.
- If they move a lot, the data are not informative enough, and the prior is doing real work. Report the dependence, think harder about the prior, or collect more data.

With little data the prior matters. As $n$ grows, the posterior mean approaches the data mean regardless of a (non-degenerate) prior. `prior_sensitivity.py` shows both effects for the 13-heads-in-20 example and for the same proportion at $n = 2000$. Saves `output/05_prior_sensitivity.png`.

## Scripts

- `conjugate.py` - Beta-Binomial and Normal-Normal updates (importable; also used by stages 06 and 07).
- `prior_sensitivity.py` - the same data under five priors at small and large $n$.

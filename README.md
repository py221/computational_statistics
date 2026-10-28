# Computational statistics: probability, estimation and Bayesian inference

A hands-on learning path from basic probability to Bayesian inference with MCMC. Each stage has a `notes.md` (the ideas and the maths)
and small runnable Python scripts (numpy, scipy, matplotlib) that check the maths numerically. A `tests/` folder asserts the key
results, such as "the numerical MLE matches the closed form" and "the Metropolis posterior mean matches the conjugate answer".

## The path

| Stage | Folder | What it covers | Key takeaway |
|---|---|---|---|
| 1 | [`01-probability`](01-probability) | Sample spaces, events, axioms, conditional probability, independence, chain rule, random variables, marginals | Everything is built from a few rules; the chain rule is what makes Bayes' rule possible |
| 2 | [`02-distributions`](02-distributions) | pmf/pdf/cdf, common families, moments, joint distributions, covariance and correlation | A distribution is summarised by its moments; zero covariance does not mean independence |
| 3 | [`03-estimation`](03-estimation) | Bias, consistency, MSE, method of moments, maximum likelihood, MAP | MLE maximises the likelihood; MAP adds a prior; MSE = variance + bias squared |
| 4 | [`04-bayes-and-decisions`](04-bayes-and-decisions) | Bayes' rule, posterior proportional to likelihood times prior, decision boundaries, loss functions | The posterior says what to believe; the loss function says what to do |
| 5 | [`05-priors`](05-priors) | Beta-Binomial and Normal-Normal conjugacy, informative versus weak priors, sensitivity | Conjugate updates are pseudo-count arithmetic; the prior matters less as data grows |
| 6 | [`06-mcmc`](06-mcmc) | Metropolis-Hastings from scratch, trace plots, acceptance rate, ESS, R-hat | Sampling replaces integration; diagnostics tell you whether to trust the samples |
| 7 | [`07-pymc`](07-pymc) | A Beta-Binomial model and a linear regression in PyMC | A library gives the same answer as the exact and hand-written results, with less work |

Read the stages in order: each one reuses ideas (and sometimes code) from the earlier ones. The running example is 13 heads in 20 coin flips.

## Layout

```
01-probability/          notes.md, dice_rules.py
02-distributions/        notes.md, distributions.py, joint_covariance.py
03-estimation/           notes.md, bernoulli_mle_map.py, mom_vs_mle.py, estimator_properties.py
04-bayes-and-decisions/  notes.md, bayes_rule.py, decision_boundary.py, posterior_summaries.py
05-priors/               notes.md, conjugate.py, prior_sensitivity.py
06-mcmc/                 notes.md, mcmc.py, mh_beta_binomial.py
07-pymc/                 notes.md, beta_binomial_pymc.py, linear_regression_pymc.py
tests/                   pytest checks for every stage
```

## How to run

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run any script from the repository root; plots are written to `output/`, which is git-ignored:

```bash
python 01-probability/dice_rules.py
python 03-estimation/bernoulli_mle_map.py
python 06-mcmc/mh_beta_binomial.py
python 07-pymc/beta_binomial_pymc.py
```

Run the tests:

```bash
python -m pytest
```

Stage 07 needs PyMC and ArviZ (included in `requirements.txt`). If they are missing, its scripts print an install hint and exit, and its tests are skipped.

## Notes on the maths

The notes use GitHub's native LaTeX rendering (`$...$` inline, `$$...$$` for display). If you read them somewhere that does not render
LaTeX, the formulas appear as plain TeX source.

## License

[MIT](LICENSE)

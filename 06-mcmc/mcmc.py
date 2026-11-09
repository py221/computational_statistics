"""Random-walk Metropolis sampler and basic MCMC diagnostics, in plain numpy."""
import numpy as np


def metropolis(log_prob, x0, n_samples, step, rng=None):
    """Random-walk Metropolis for a (possibly unnormalised) log density.

    Parameters
    ----------
    log_prob : callable
        Maps a 1-D numpy array (the parameter vector) to the log of the unnormalised
        target density. Return -inf for impossible values.
    x0 : float or array
        Starting point.
    n_samples : int
        Number of iterations (the starting point is not included).
    step : float or array
        Standard deviation of the Gaussian proposal.
    rng : numpy.random.Generator, optional

    Returns
    -------
    samples : ndarray, shape (n_samples, dim)
    acceptance_rate : float
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_1d(np.asarray(x0, dtype=float))
    step = np.broadcast_to(np.asarray(step, dtype=float), x.shape)
    current = log_prob(x)
    if not np.isfinite(current):
        raise ValueError("log_prob(x0) must be finite")

    samples = np.empty((n_samples, x.size))
    accepted = 0
    for t in range(n_samples):
        proposal = x + step * rng.standard_normal(x.size)
        proposed = log_prob(proposal)
        # symmetric proposal: accept with probability min(1, p(prop)/p(current))
        if np.log(rng.uniform()) < proposed - current:
            x, current = proposal, proposed
            accepted += 1
        samples[t] = x
    return samples, accepted / n_samples


def run_chains(log_prob, starts, n_samples, step, seed=0):
    """Run one chain per starting point. Returns (chains, acceptance_rates).

    chains has shape (n_chains, n_samples, dim).
    """
    seeds = np.random.SeedSequence(seed).spawn(len(starts))
    out, rates = [], []
    for s, x0 in zip(seeds, starts):
        c, r = metropolis(log_prob, x0, n_samples, step, np.random.default_rng(s))
        out.append(c)
        rates.append(r)
    return np.stack(out), np.array(rates)


def autocorrelation(x, max_lag=None):
    """Sample autocorrelation of a 1-D series for lags 0..max_lag (FFT based)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[: max_lag + 1] / n
    if acov[0] == 0:
        return np.ones(max_lag + 1)
    return acov / acov[0]


def effective_sample_size(x, max_lag=None):
    """ESS = n / (1 + 2 * sum rho_k), summing until the first non-positive pair of lags."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rho = autocorrelation(x, max_lag)
    total = 0.0
    for k in range(1, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        total += pair
    tau = 1.0 + 2.0 * total
    return n / tau


def split_rhat(chains):
    """Split R-hat for chains of shape (n_chains, n_draws) (one parameter)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    half = n // 2
    split = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    n_half = split.shape[1]
    chain_means = split.mean(axis=1)
    W = split.var(axis=1, ddof=1).mean()
    B = n_half * chain_means.var(ddof=1)
    var_plus = (n_half - 1) / n_half * W + B / n_half
    return float(np.sqrt(var_plus / W))

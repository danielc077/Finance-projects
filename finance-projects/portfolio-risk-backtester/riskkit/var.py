"""One-day Value at Risk and Expected Shortfall. Results are positive loss fractions."""
import numpy as np
from scipy.stats import norm


def historical_var_es(port_returns, alpha=0.99):
    r = np.asarray(port_returns)
    var = -np.quantile(r, 1 - alpha)
    tail = r[r <= -var]
    es = -tail.mean() if len(tail) else var
    return float(var), float(es)


def parametric_var_es(port_returns, alpha=0.99):
    """Gaussian (variance-covariance) VaR/ES."""
    r = np.asarray(port_returns)
    mu, sd = r.mean(), r.std(ddof=1)
    z = norm.ppf(alpha)
    var = -(mu - z * sd)
    es = -(mu - sd * norm.pdf(z) / (1 - alpha))
    return float(var), float(es)


def monte_carlo_var_es(asset_returns, weights, alpha=0.99, n_sims=100_000, seed=0):
    """Simulate from a multivariate normal fitted to the asset return window."""
    rng = np.random.default_rng(seed)
    mu, cov = asset_returns.mean().values, asset_returns.cov().values
    sims = rng.multivariate_normal(mu, cov, size=n_sims) @ np.asarray(weights)
    return historical_var_es(sims, alpha)

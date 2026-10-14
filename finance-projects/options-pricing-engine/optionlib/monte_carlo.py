"""Monte Carlo pricers under GBM with antithetic variates and standard-error reporting."""
import numpy as np


def mc_european(S, K, T, r, sigma, option="call", q=0.0, n_paths=200_000,
                antithetic=True, seed=None):
    """Return (price, standard_error). With antithetic=True, n_paths is the total path count."""
    rng = np.random.default_rng(seed)
    sign = 1.0 if option == "call" else -1.0
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        pay = 0.5 * (np.maximum(sign * (S * np.exp(drift + vol * z) - K), 0)
                     + np.maximum(sign * (S * np.exp(drift - vol * z) - K), 0))
    else:
        z = rng.standard_normal(n_paths)
        pay = np.maximum(sign * (S * np.exp(drift + vol * z) - K), 0)
    disc = np.exp(-r * T)
    return float(disc * pay.mean()), float(disc * pay.std(ddof=1) / np.sqrt(len(pay)))


def mc_barrier_down_and_out_call(S, K, H, T, r, sigma, q=0.0, n_paths=100_000,
                                 n_steps=252, seed=None):
    """Down-and-out call (discretely monitored). Returns (price, standard_error)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_paths = np.log(S) + np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    alive = (log_paths > np.log(H)).all(axis=1)
    pay = np.where(alive, np.maximum(np.exp(log_paths[:, -1]) - K, 0.0), 0.0)
    disc = np.exp(-r * T)
    return float(disc * pay.mean()), float(disc * pay.std(ddof=1) / np.sqrt(n_paths))

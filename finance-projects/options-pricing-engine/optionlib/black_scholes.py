"""Closed-form Black-Scholes-Merton pricing and Greeks (continuous dividend yield q)."""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """European option price. option in {'call','put'}."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option="call", q=0.0):
    """Return dict of delta, gamma, vega (per 1.00 vol), theta (per year), rho (per 1.00 rate)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * np.sqrt(T))
    vega = S * disc_q * pdf * np.sqrt(T)
    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

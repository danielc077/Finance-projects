"""Implied volatility: Newton-Raphson with a bisection fallback."""
import numpy as np
from .black_scholes import bs_price, bs_greeks


def implied_vol(price, S, K, T, r, option="call", q=0.0, tol=1e-8, max_iter=100):
    """Solve bs_price(sigma) = price. Returns nan if the price violates no-arbitrage bounds."""
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if option == "call":
        lo_bound, hi_bound = max(S * disc_q - K * disc_r, 0.0), S * disc_q
    else:
        lo_bound, hi_bound = max(K * disc_r - S * disc_q, 0.0), K * disc_r
    if not (lo_bound < price < hi_bound):
        return float("nan")

    sigma = 0.2
    for _ in range(max_iter):                      # Newton
        diff = bs_price(S, K, T, r, sigma, option, q) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = bs_greeks(S, K, T, r, sigma, option, q)["vega"]
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if not (1e-6 < sigma < 10):
            break

    lo, hi = 1e-6, 10.0                            # bisection fallback
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, option, q) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))

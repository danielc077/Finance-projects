"""Cox-Ross-Rubinstein binomial tree for European and American options (vectorised)."""
import numpy as np


def binomial_price(S, K, T, r, sigma, option="call", q=0.0, steps=500, american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sign = 1.0 if option == "call" else -1.0

    j = np.arange(steps + 1)
    prices = S * u ** (steps - j) * d ** j          # terminal asset prices
    values = np.maximum(sign * (prices - K), 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            prices = S * u ** (i - j) * d ** j
            values = np.maximum(values, sign * (prices - K))
    return float(values[0])

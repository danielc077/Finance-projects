"""Demo: model cross-checks, Greeks table, MC convergence, and implied-vol smile.

Run:  python demo.py
Outputs PNG charts into ./outputs/

Smile note: with no market data feed, option prices are generated from a skewed vol
function and then *inverted* with the implied-vol solver, which verifies the solver
and shows the smile shape. To use real quotes, replace `market_quotes()` with
rows of (strike, mid_price) from your data source (e.g. yfinance option_chain).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optionlib import (bs_price, bs_greeks, binomial_price, mc_european,
                       mc_barrier_down_and_out_call, implied_vol)

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.04, 0.25, 0.01


def compare_models():
    print("=== European call: model comparison ===")
    bs = bs_price(S, K, T, r, sigma, "call", q)
    bino = binomial_price(S, K, T, r, sigma, "call", q, steps=1000)
    mc, se = mc_european(S, K, T, r, sigma, "call", q, n_paths=400_000, seed=42)
    print(f"Black-Scholes : {bs:.4f}")
    print(f"Binomial 1000 : {bino:.4f}  (diff {bino - bs:+.4f})")
    print(f"Monte Carlo   : {mc:.4f} +/- {1.96 * se:.4f} (95% CI)")
    am = binomial_price(S, K, T, r, sigma, "put", q, steps=1000, american=True)
    eu = bs_price(S, K, T, r, sigma, "put", q)
    print(f"American put  : {am:.4f} vs European put {eu:.4f} (early-exercise premium {am - eu:.4f})")
    bar, bse = mc_barrier_down_and_out_call(S, K, 85, T, r, sigma, q, seed=7)
    print(f"Down-and-out call (H=85): {bar:.4f} +/- {1.96 * bse:.4f}")


def greeks_table():
    print("\n=== Greeks (call) ===")
    for k, v in bs_greeks(S, K, T, r, sigma, "call", q).items():
        print(f"{k:>6}: {v:+.5f}")


def convergence_plot():
    bs = bs_price(S, K, T, r, sigma, "call", q)
    ns = np.unique(np.logspace(3, 5.7, 12).astype(int))
    plain, anti = [], []
    for n in ns:
        plain.append(abs(mc_european(S, K, T, r, sigma, "call", q, n, antithetic=False, seed=1)[0] - bs))
        anti.append(abs(mc_european(S, K, T, r, sigma, "call", q, n, antithetic=True, seed=1)[0] - bs))
    steps = [10, 20, 50, 100, 200, 500, 1000]
    tree = [abs(binomial_price(S, K, T, r, sigma, "call", q, s) - bs) for s in steps]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(ns, plain, "o-", label="Plain MC")
    ax[0].loglog(ns, anti, "s-", label="Antithetic MC")
    ax[0].set_xlabel("paths"); ax[0].set_ylabel("|price - Black-Scholes|"); ax[0].legend()
    ax[0].set_title("Monte Carlo convergence")
    ax[1].loglog(steps, tree, "o-", color="C2")
    ax[1].set_xlabel("tree steps"); ax[1].set_title("Binomial convergence")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "convergence.png"), dpi=150); plt.close()


def market_quotes():
    """Synthetic quotes with a realistic skew; swap for real data."""
    strikes = np.linspace(70, 130, 25)
    true_vol = 0.20 + 0.0006 * (strikes - 100) ** 2 / 10 - 0.002 * (strikes - 100)
    prices = [bs_price(S, k, 0.5, r, v, "call", q) for k, v in zip(strikes, true_vol)]
    return strikes, np.array(prices)


def smile_plot():
    strikes, prices = market_quotes()
    ivs = [implied_vol(p, S, k, 0.5, r, "call", q) for k, p in zip(strikes, prices)]
    plt.figure(figsize=(6, 4))
    plt.plot(strikes, np.array(ivs) * 100, "o-")
    plt.axvline(S, ls="--", color="grey")
    plt.xlabel("Strike"); plt.ylabel("Implied vol (%)"); plt.title("Implied volatility smile")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "vol_smile.png"), dpi=150); plt.close()


if __name__ == "__main__":
    compare_models()
    greeks_table()
    convergence_plot()
    smile_plot()
    print(f"\nCharts saved to {OUT}")

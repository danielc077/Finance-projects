"""End-to-end risk report.

Run:  python run_analysis.py                  (live data via yfinance, else synthetic fallback)
      python run_analysis.py --csv prices.csv (your own Date x ticker price file)
Outputs PNG charts into ./outputs/
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from riskkit import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def main(csv=None):
    prices, source = load_prices(csv=csv)
    rets = prices.pct_change().dropna()
    n = rets.shape[1]
    w = np.full(n, 1 / n)
    port = rets @ w
    print(f"Data source: {source} | {len(rets)} days | {n} assets "
          f"({rets.index[0].date()} to {rets.index[-1].date()})")
    if source == "synthetic":
        print("NOTE: synthetic data - results demonstrate methodology, not market performance.\n")

    print("=== Asset statistics (annualised) ===")
    print(annualised_stats(rets).round(3).to_string())

    # ---- VaR / ES, full sample ----
    print("\n=== 1-day VaR / ES (equal-weight portfolio, full sample) ===")
    for a in (0.95, 0.99):
        for name, fn in [("Historical", lambda: historical_var_es(port, a)),
                         ("Parametric", lambda: parametric_var_es(port, a)),
                         ("Monte Carlo", lambda: monte_carlo_var_es(rets, w, a))]:
            v, e = fn()
            print(f"{a:.0%} {name:<11} VaR {v:6.2%}  ES {e:6.2%}")

    # ---- Rolling backtest + tests ----
    print("\n=== 99% VaR backtest (500-day rolling window) ===")
    bt = {}
    for m in ("historical", "parametric", "montecarlo"):
        bt[m] = rolling_var_backtest(rets, w, m, 0.99, window=500, step=1 if m != "montecarlo" else 5)
        k = kupiec_pof_test(bt[m]["breach"], 0.99)
        c = christoffersen_independence_test(bt[m]["breach"])
        verdict = "PASS" if k["p_value"] > 0.05 else "REJECT"
        print(f"{m:<11} breaches {k['breaches']:>3}/{k['observations']} (expected {k['expected']:.1f}) | "
              f"Kupiec p={k['p_value']:.3f} [{verdict}] | Christoffersen p={c['p_value']:.3f}")

    h = bt["historical"]
    plt.figure(figsize=(11, 4))
    plt.plot(h.index, h["realised"] * 100, lw=0.6, label="Daily return")
    plt.plot(h.index, -h["var"] * 100, color="red", lw=1, label="-99% VaR (historical)")
    plt.scatter(h.index[h["breach"]], h["realised"][h["breach"]] * 100, color="black", s=14, zorder=3, label="Breach")
    plt.ylabel("%"); plt.title("VaR backtest"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "var_backtest.png"), dpi=150); plt.close()

    # ---- Optimisation ----
    print("\n=== Optimised portfolios (long-only) ===")
    for name, wt in [("Min variance", min_variance(rets)), ("Max Sharpe", max_sharpe(rets))]:
        r = rets @ wt
        print(f"{name:<13} Sharpe {sharpe_ratio(r):.2f} | vol {r.std() * np.sqrt(252):.1%} | "
              + ", ".join(f"{t} {x:.0%}" for t, x in zip(rets.columns, wt) if x > 0.01))
    fr = efficient_frontier(rets)
    ann = annualised_stats(rets)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fr[:, 0] * 100, fr[:, 1] * 100, "-o", ms=3, label="Efficient frontier")
    plt.scatter(ann["ann_vol"] * 100, ann["ann_return"] * 100, c="grey", label="Assets")
    plt.xlabel("Volatility (%)"); plt.ylabel("Return (%)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "efficient_frontier.png"), dpi=150); plt.close()

    # ---- Strategy backtest ----
    print("\n=== Momentum strategy (top 3 by 6-month return, monthly rebalance, 10 bps cost) ===")
    net, bench, stats = momentum_backtest(prices)
    for k_, v in stats.items():
        print(f"{k_:<32} {v:8.3f}")
    plt.figure(figsize=(9, 4))
    (1 + net).cumprod().plot(label="Momentum (net of costs)")
    (1 + bench).cumprod().plot(label="Equal-weight benchmark")
    plt.legend(); plt.title("Cumulative growth of $1"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "momentum_backtest.png"), dpi=150); plt.close()
    print(f"\nCharts saved to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv")
    main(ap.parse_args().csv)

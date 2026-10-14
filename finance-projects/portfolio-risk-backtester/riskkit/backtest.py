"""VaR backtesting (Kupiec & Christoffersen tests) and a momentum strategy backtest."""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .var import historical_var_es, parametric_var_es, monte_carlo_var_es
from .metrics import sharpe_ratio, max_drawdown, TRADING_DAYS


def rolling_var_backtest(asset_returns, weights, method="historical", alpha=0.99,
                         window=500, step=1, mc_sims=20_000):
    """For each day, estimate VaR from the prior `window` days and compare with the realised return.

    Returns DataFrame[var, realised, breach].
    """
    w = np.asarray(weights)
    port = asset_returns @ w
    rows = []
    for t in range(window, len(port), step):
        hist = port.iloc[t - window:t]
        if method == "historical":
            var, _ = historical_var_es(hist, alpha)
        elif method == "parametric":
            var, _ = parametric_var_es(hist, alpha)
        elif method == "montecarlo":
            var, _ = monte_carlo_var_es(asset_returns.iloc[t - window:t], w, alpha, mc_sims, seed=t)
        else:
            raise ValueError(method)
        rows.append((port.index[t], var, port.iloc[t], port.iloc[t] < -var))
    return pd.DataFrame(rows, columns=["date", "var", "realised", "breach"]).set_index("date")


def kupiec_pof_test(breaches, alpha=0.99):
    """Proportion-of-failures likelihood-ratio test. H0: breach rate == 1-alpha."""
    b = np.asarray(breaches, dtype=bool)
    n, x, p = len(b), int(b.sum()), 1 - alpha
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return {"observations": n, "breaches": x, "expected": n * p,
            "LR": float(lr), "p_value": float(1 - chi2.cdf(lr, 1))}


def christoffersen_independence_test(breaches):
    """Tests whether breaches cluster (violating independence). H0: independent."""
    b = np.asarray(breaches, dtype=int)
    a, c = b[:-1], b[1:]
    n00 = int(((a == 0) & (c == 0)).sum()); n01 = int(((a == 0) & (c == 1)).sum())
    n10 = int(((a == 1) & (c == 0)).sum()); n11 = int(((a == 1) & (c == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, k, m):   # k successes of m, safe at 0/1
        return (k * np.log(p) if k else 0) + ((m - k) * np.log(1 - p) if m - k else 0)

    l0 = ll(pi, n01 + n11, n00 + n01 + n10 + n11)
    l1 = ll(pi01, n01, n00 + n01) + ll(pi11, n11, n10 + n11)
    lr = max(-2 * (l0 - l1), 0.0)
    return {"LR": float(lr), "p_value": float(1 - chi2.cdf(lr, 1))}


def momentum_backtest(prices, lookback=126, top_n=3, rebalance_days=21, cost_bps=10):
    """Long-only cross-sectional momentum: hold the top_n trailing-return assets, equal weight.

    Signals use data up to t-1 and are applied to day t returns (no look-ahead).
    Transaction cost = cost_bps per unit of turnover. Returns (net_returns, stats dict).
    """
    rets = prices.pct_change().dropna()
    mom = prices.pct_change(lookback).shift(1).reindex(rets.index)
    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    cur = pd.Series(0.0, index=rets.columns)
    for i, dt in enumerate(rets.index):
        if i >= lookback and (i - lookback) % rebalance_days == 0:
            winners = mom.loc[dt].nlargest(top_n).index
            cur = pd.Series(0.0, index=rets.columns); cur[winners] = 1 / top_n
        weights.loc[dt] = cur
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    net = (gross - turnover * cost_bps / 1e4).iloc[lookback:]
    bench = rets.mean(axis=1).iloc[lookback:]
    stats = {
        "ann_return": float(net.mean() * TRADING_DAYS),
        "ann_vol": float(net.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe_ratio(net), "max_drawdown": max_drawdown(net),
        "benchmark_sharpe(equal-weight)": sharpe_ratio(bench),
        "avg_annual_turnover": float(turnover.sum() / len(turnover) * TRADING_DAYS),
    }
    return net, bench, stats

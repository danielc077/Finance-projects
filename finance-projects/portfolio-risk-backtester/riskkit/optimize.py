"""Long-only mean-variance optimisation."""
import numpy as np
from scipy.optimize import minimize
from .metrics import TRADING_DAYS


def _setup(returns):
    return returns.mean().values * TRADING_DAYS, returns.cov().values * TRADING_DAYS


def _solve(obj, n, extra_cons=()):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_cons]
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n, constraints=cons)
    return res.x


def min_variance(returns):
    _, cov = _setup(returns)
    return _solve(lambda w: w @ cov @ w, len(cov))


def max_sharpe(returns, rf=0.0):
    mu, cov = _setup(returns)
    return _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), len(mu))


def efficient_frontier(returns, n_points=30):
    mu, cov = _setup(returns)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    out = []
    for tgt in targets:
        w = _solve(lambda w: w @ cov @ w, len(mu),
                   extra_cons=[{"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}])
        out.append((float(np.sqrt(w @ cov @ w)), float(w @ mu)))
    return np.array(out)

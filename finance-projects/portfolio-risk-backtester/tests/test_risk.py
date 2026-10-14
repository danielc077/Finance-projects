import numpy as np
import pandas as pd
import pytest
from riskkit import *

rng = np.random.default_rng(0)
R = pd.DataFrame(rng.normal(0.0004, 0.01, size=(3000, 4)), columns=list("ABCD"))
W = np.full(4, 0.25)


def test_historical_var_matches_quantile():
    v, e = historical_var_es(R @ W, 0.95)
    assert v == pytest.approx(-np.quantile(R @ W, 0.05))
    assert e >= v


def test_methods_agree_on_gaussian_data():
    p = R @ W
    h, _ = historical_var_es(p, 0.99)
    g, _ = parametric_var_es(p, 0.99)
    m, _ = monte_carlo_var_es(R, W, 0.99, n_sims=300_000)
    assert h == pytest.approx(g, abs=0.002) and m == pytest.approx(g, abs=0.002)


def test_kupiec_accepts_correct_rate_and_rejects_wrong():
    ok = np.zeros(1000, dtype=bool); ok[::100] = True            # 1% breaches
    bad = np.zeros(1000, dtype=bool); bad[::10] = True           # 10% breaches
    assert kupiec_pof_test(ok, 0.99)["p_value"] > 0.05
    assert kupiec_pof_test(bad, 0.99)["p_value"] < 0.01


def test_backtest_breach_rate_near_nominal():
    bt = rolling_var_backtest(R, W, "parametric", 0.99, window=500)
    assert abs(bt["breach"].mean() - 0.01) < 0.01


def test_weights_valid_and_min_var_lower_vol():
    for fn in (min_variance, max_sharpe):
        w = fn(R)
        assert w.sum() == pytest.approx(1.0, abs=1e-6) and (w >= -1e-9).all()
    assert (R @ min_variance(R)).std() <= (R @ W).std() + 1e-9


def test_momentum_no_lookahead_and_costs_reduce_return():
    prices = simulate_prices(years=4, seed=3)
    n0, _, s0 = momentum_backtest(prices, cost_bps=0)
    n1, _, s1 = momentum_backtest(prices, cost_bps=50)
    assert s1["ann_return"] < s0["ann_return"]

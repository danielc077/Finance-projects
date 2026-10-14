import math
import numpy as np
import pytest
from optionlib import (bs_price, bs_greeks, binomial_price, mc_european,
                       mc_barrier_down_and_out_call, implied_vol)

S, K, T, r, sigma = 100, 105, 1.0, 0.03, 0.25


def test_textbook_value():
    # Hull-style check value for S=100,K=100,T=1,r=5%,sigma=20% -> 10.4506
    assert bs_price(100, 100, 1, 0.05, 0.20, "call") == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    c, p = bs_price(S, K, T, r, sigma, "call"), bs_price(S, K, T, r, sigma, "put")
    assert c - p == pytest.approx(S - K * math.exp(-r * T), abs=1e-10)


def test_binomial_converges_to_bs():
    assert binomial_price(S, K, T, r, sigma, "call", steps=2000) == pytest.approx(
        bs_price(S, K, T, r, sigma, "call"), abs=5e-3)


def test_american_put_ge_european():
    am = binomial_price(S, K, T, r, sigma, "put", steps=500, american=True)
    eu = binomial_price(S, K, T, r, sigma, "put", steps=500)
    assert am >= eu


def test_monte_carlo_within_error():
    px, se = mc_european(S, K, T, r, sigma, "call", n_paths=400_000, seed=1)
    assert abs(px - bs_price(S, K, T, r, sigma, "call")) < 4 * se


def test_barrier_below_vanilla():
    b, _ = mc_barrier_down_and_out_call(S, K, 85, T, r, sigma, n_paths=50_000, seed=2)
    assert b < bs_price(S, K, T, r, sigma, "call")


def test_greeks_match_finite_difference():
    h = 1e-4
    g = bs_greeks(S, K, T, r, sigma, "call")
    fd_delta = (bs_price(S + h, K, T, r, sigma) - bs_price(S - h, K, T, r, sigma)) / (2 * h)
    fd_vega = (bs_price(S, K, T, r, sigma + h) - bs_price(S, K, T, r, sigma - h)) / (2 * h)
    assert g["delta"] == pytest.approx(fd_delta, abs=1e-6)
    assert g["vega"] == pytest.approx(fd_vega, abs=1e-4)


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("true_vol", [0.08, 0.25, 0.9])
def test_implied_vol_roundtrip(opt, true_vol):
    px = bs_price(S, K, T, r, true_vol, opt)
    assert implied_vol(px, S, K, T, r, opt) == pytest.approx(true_vol, abs=1e-6)


def test_implied_vol_rejects_arbitrage_price():
    assert np.isnan(implied_vol(0.0, S, K, T, r, "call"))

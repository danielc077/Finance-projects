# Options Pricing & Greeks Engine

A from-scratch derivatives pricing library in Python, with four independent
pricing methods that cross-validate each other.

## Features
- **Black-Scholes-Merton** closed form with continuous dividend yield, plus all five Greeks (delta, gamma, vega, theta, rho)
- **CRR binomial tree** for European and **American** options (vectorised, early-exercise premium)
- **Monte Carlo** under GBM with antithetic variance reduction and standard-error reporting
- **Path-dependent pricing**: discretely monitored down-and-out barrier call
- **Implied volatility solver**: Newton-Raphson with bisection fallback and no-arbitrage bound checks
- Test suite: put-call parity, textbook values, Greeks vs finite differences, model convergence, IV round-trips

## Run
```bash
pip install -r requirements.txt
python demo.py        # prints model comparison + Greeks, saves charts to outputs/
pytest -q             # run tests
```

## Example output
```
Black-Scholes : 9.0310
Binomial 1000 : 9.0300  (diff -0.0010)
Monte Carlo   : 9.0598 +/- 0.0419 (95% CI)
American put  : 11.3613 vs European put 10.9089 (early-exercise premium 0.4524)
Down-and-out call (H=85): 8.2143 +/- 0.0980
```

## Design notes
- Antithetic variates roughly halve MC variance for monotone payoffs.
- Barrier pricing is discretely monitored, so it slightly overprices vs continuous monitoring (known discretisation bias).
- Vol smile in `demo.py` uses synthetic quotes to validate the solver; plug real option chains into `market_quotes()` for live data.

## Possible extensions
Heston stochastic vol, local vol (Dupire), vol surface interpolation (SVI), finite-difference PDE solver.

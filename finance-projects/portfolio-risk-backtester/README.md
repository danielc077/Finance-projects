# Portfolio Risk & Backtesting Toolkit

A risk-management toolkit: Value at Risk / Expected Shortfall three ways, statistical
backtesting of those models, mean-variance optimisation, and a cost-aware strategy backtest.

## Features
- **VaR & ES** at 95%/99%: historical simulation, parametric (variance-covariance), Monte Carlo
- **Model validation**: rolling-window VaR backtest with the **Kupiec POF test** (breach frequency) and **Christoffersen independence test** (breach clustering)
- **Optimisation**: long-only minimum-variance, max-Sharpe and efficient frontier (SciPy SLSQP)
- **Strategy backtest**: cross-sectional momentum with monthly rebalancing, no look-ahead (signals lagged one day), transaction costs, Sharpe, max drawdown, turnover vs equal-weight benchmark
- Data layer: live data via `yfinance`, your own CSV, or a reproducible synthetic fallback (fat tails + volatility clustering)

## Run
```bash
pip install -r requirements.txt
python run_analysis.py                    # live data if reachable, else synthetic
python run_analysis.py --csv prices.csv   # your own prices (Date index, one column per ticker)
pytest -q
```

## Sample output (synthetic data; rerun with real data for your resume numbers)
```
99% Historical  VaR  3.05%  ES  3.69%
historical  breaches  28/2019 (expected 20.2) | Kupiec p=0.099 [PASS] | Christoffersen p=0.375
Max Sharpe    Sharpe 1.01 | vol 21.4% | MSFT 29%, JNJ 55%, ...
```

## Design notes / limitations
- Parametric and Monte Carlo VaR assume normality, so they understate tail risk on fat-tailed data (compare 99% ES across methods).
- The synthetic fallback exists so the project runs offline; any figures you quote should come from real market data.
- Max-Sharpe weights are in-sample and sensitive to estimation error; a walk-forward or shrinkage-covariance extension would address this.

## Possible extensions
GARCH/EWMA volatility for VaR, filtered historical simulation, stress testing (2008/2020 scenarios), Black-Litterman, factor risk decomposition.

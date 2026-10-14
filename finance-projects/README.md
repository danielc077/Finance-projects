# Finance Projects

1. `options-pricing-engine/` - derivatives pricing (Black-Scholes, binomial, Monte Carlo), Greeks, implied vol
2. `portfolio-risk-backtester/` - VaR/ES, backtesting tests, portfolio optimisation, strategy backtest

Each folder has its own README. Python 3.9+; `pip install -r requirements.txt` then run the demo/analysis script.

## Resume bullets (fill in real numbers after running on real data)
**Options Pricing & Greeks Engine** | Python, NumPy, SciPy
- Built a derivatives pricing library implementing Black-Scholes, CRR binomial (European/American) and Monte Carlo (antithetic variates, barrier options), cross-validated to within 0.01 of closed-form prices
- Implemented Greeks and a Newton-Raphson/bisection implied-volatility solver; verified against finite differences with a 14-case automated test suite

**Portfolio Risk & Backtesting Toolkit** | Python, pandas, SciPy
- Developed a risk framework computing 1-day VaR/ES (historical, parametric, Monte Carlo) and validated models with rolling backtests using Kupiec and Christoffersen tests
- Built long-only mean-variance optimiser and a momentum strategy backtester with transaction costs and look-ahead protection, reporting Sharpe, drawdown and turnover

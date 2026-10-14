import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualised_stats(returns: pd.DataFrame):
    return pd.DataFrame({"ann_return": returns.mean() * TRADING_DAYS,
                         "ann_vol": returns.std() * np.sqrt(TRADING_DAYS)})


def sharpe_ratio(returns, rf=0.0):
    ex = returns - rf / TRADING_DAYS
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns):
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())

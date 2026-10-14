"""Price loading: yfinance if available, else a reproducible synthetic fallback."""
import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "GS", "NVDA"]


def simulate_prices(tickers=DEFAULT_TICKERS, years=10, seed=0):
    """Correlated fat-tailed returns with volatility clustering (GARCH-like), ~business-day index.

    Intended as a stand-in when no market data is reachable. Results on synthetic
    data demonstrate the methodology, not real-market performance.
    """
    rng = np.random.default_rng(seed)
    n, k = int(252 * years), len(tickers)
    A = rng.normal(size=(k, k)); corr = A @ A.T
    d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
    corr = 0.35 + 0.65 * corr * 0.5 + 0.15 * np.eye(k)   # keep positive definite, moderate correlation
    d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
    L = np.linalg.cholesky(corr)
    base_vol = rng.uniform(0.18, 0.40, k) / np.sqrt(252)
    mu = rng.uniform(0.04, 0.14, k) / 252
    nu = 5
    z = (rng.standard_t(nu, size=(n, k)) / np.sqrt(nu / (nu - 2))) @ L.T
    rets = np.zeros((n, k)); var_mult = np.ones(k)
    for t in range(n):                                   # volatility clustering
        rets[t] = mu + base_vol * np.sqrt(var_mult) * z[t]
        shock = (rets[t] - mu) ** 2 / base_vol**2
        var_mult = 0.05 + 0.10 * shock + 0.85 * var_mult
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


def load_prices(tickers=DEFAULT_TICKERS, years=10, csv=None, allow_synthetic=True):
    """Return (prices DataFrame, source string). csv: path with Date index + one column per ticker."""
    if csv:
        return pd.read_csv(csv, index_col=0, parse_dates=True).dropna(), f"csv:{csv}"
    try:
        import yfinance as yf
        df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"].dropna()
        if len(df) > 250:
            return df, "yfinance"
    except Exception:
        pass
    if not allow_synthetic:
        raise RuntimeError("Could not download data and synthetic fallback disabled.")
    return simulate_prices(tickers, years), "synthetic"

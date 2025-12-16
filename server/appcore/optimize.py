import numpy as np
import pandas as pd


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Compute portfolio weights that maximize the Sharpe ratio."""
    tickers = list(mu.index)
    mu = mu.values.reshape(-1, 1)
    inv_cov = np.linalg.pinv(cov.values)
    ones = np.ones((len(tickers), 1))
    excess = mu - risk_free
    w = inv_cov @ excess
    w /= ones.T @ w  # normalize
    w = w.flatten()
    return pd.Series(w, index=tickers)

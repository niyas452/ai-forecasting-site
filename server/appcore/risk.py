import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf


def ewma_vol(returns: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """EWMA volatility (last row) from returns DataFrame (rows=time, cols=tickers)."""
    var = returns.ewm(alpha=1 - lam).var()
    return var.iloc[-1].pow(0.5)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf shrinkage covariance on returns."""
    # drop rows with any NaNs
    X = returns.dropna().values
    lw = LedoitWolf().fit(X)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return cov

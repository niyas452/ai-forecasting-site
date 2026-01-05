# appcore/risk.py
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def ewma_vol(returns: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    var = returns.ewm(alpha=1 - lam).var()
    return var.iloc[-1].pow(0.5)

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    
    if returns is None or returns.empty:
        return pd.DataFrame()

    R = returns.copy()
    R = R.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if R.empty:
        return pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)

    lw = LedoitWolf().fit(R.values)
    cov = pd.DataFrame(lw.covariance_, index=R.columns, columns=R.columns)
    return cov

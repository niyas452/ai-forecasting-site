import numpy as np
import pandas as pd

MONTHS = {"6m": 6, "12m": 12}


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    
    if prices is None or prices.empty:
        return pd.DataFrame()

    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.replace([np.inf, -np.inf], np.nan)

    out = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
       
        out[col] = s.resample("ME").last()

    if not out:
        return pd.DataFrame()

    m = pd.concat(out, axis=1)
    return m.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
   
    if prices is None or prices.empty:
        return pd.DataFrame()
    p = prices.replace([np.inf, -np.inf], np.nan)
    return np.log(p / p.shift(1))


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    
    if prices is None or prices.empty:
        return pd.DataFrame()
    p = prices.replace([np.inf, -np.inf], np.nan)
    return p.pct_change()

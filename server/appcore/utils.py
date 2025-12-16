import numpy as np
import pandas as pd

MONTHS = {"6m": 6, "12m": 12}


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices to month-end prices."""
    m = prices.resample("ME").last()  # 'M' deprecated in pandas 2.2+
    return m.ffill().dropna(how="all")




def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly percent returns."""
    return prices.pct_change().dropna()

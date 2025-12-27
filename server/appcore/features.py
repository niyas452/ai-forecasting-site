import numpy as np
import pandas as pd
from appcore.utils import to_monthly, log_returns, MONTHS


def make_monthly_features(prices: pd.DataFrame) -> pd.DataFrame:
    m = to_monthly(prices)
    rets = log_returns(m)
    feats = {}

    # momentum
    for k in [1, 3, 6, 12]:
        feats[f"mom_{k}m"] = rets.rolling(k).sum()

    # volatility
    feats["vol_6m"] = rets.rolling(6).std()
    feats["vol_12m"] = rets.rolling(12).std()

    # skewness & kurtosis
    feats["skew_12m"] = rets.rolling(12).skew()
    feats["kurt_12m"] = rets.rolling(12).kurt()

    # moving-average gap
    ma6 = m.rolling(6).mean()
    ma12 = m.rolling(12).mean()
    feats["ma_gap"] = (m - ma6) / ma12

    X = pd.concat(feats, axis=1)
    X.columns = [f"{f}_{t}" for f, t in X.columns]  # flatten (feature, ticker)

    return X.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def targets(prices: pd.DataFrame, horizon: str) -> pd.DataFrame:
    m = to_monthly(prices)
    h = MONTHS[horizon]

    lr = np.log(m).diff()
    y = lr.shift(-h).rolling(h).sum()

    
    return y.replace([np.inf, -np.inf], np.nan).dropna(how="all")

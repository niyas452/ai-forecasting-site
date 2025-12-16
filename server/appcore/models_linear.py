import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNet


class ElasticNetModel:
    """
    Multi-task ElasticNet:
    - X: features for all tickers (columns like mom_1m_AAPL, mom_1m_MSFT, ...)
    - y: DataFrame with one column per ticker (AAPL, MSFT, SPY, ...)
    """

    def __init__(self, alpha=0.5, l1_ratio=0.5, random_state=42):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.model = None
        self.tickers: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        # align on dates
        common = X.index.intersection(y.index)
        Xc = X.loc[common]
        yc = y.loc[common]

        self.tickers = list(yc.columns)

        self.model = make_pipeline(
            StandardScaler(),
            MultiTaskElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
            ),
        )
        # Multi-task: y is (n_samples, n_tickers)
        self.model.fit(Xc, yc.values)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Return a Series indexed by ticker with the LAST time-step forecast.
        """
        if self.model is None or self.tickers is None:
            raise RuntimeError("ElasticNetModel must be fit before predict().")

        yhat = self.model.predict(X)  # shape: (n_samples, n_tickers)
        last = yhat[-1, :]            # last row = latest forecast
        return pd.Series(last, index=self.tickers)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class LGBMQuantile:
    """
    Quantile LightGBM, trained ONE model per ticker.
    predict(X) -> Series indexed by ticker (last time step).
    """

    def __init__(self, quantile: float = 0.5, random_state: int = 42):
        self.quantile = quantile
        self.random_state = random_state
        self.models: dict[str, LGBMRegressor] = {}
        self.tickers: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        common = X.index.intersection(y.index)
        Xc = X.loc[common]
        yc = y.loc[common]

        self.tickers = list(yc.columns)
        self.models = {}

        for t in self.tickers:
            # pick only features ending with _TICKER
            cols = [c for c in Xc.columns if c.endswith(f"_{t}")]
            Xi = Xc[cols]
            yi = yc[t]

            mdl = LGBMRegressor(
                objective="quantile",
                alpha=self.quantile,
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbose=-1,  # <--- mute LightGBM logs
            )
            mdl.fit(Xi, yi)
            self.models[t] = mdl

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.models:
            raise RuntimeError("LGBMQuantile must be fit before predict().")

        preds = {}
        for t, mdl in self.models.items():
            cols = [c for c in X.columns if c.endswith(f"_{t}")]
            Xi = X[cols]
            yhat = mdl.predict(Xi)
            preds[t] = yhat[-1]  # last time step

        return pd.Series(preds)

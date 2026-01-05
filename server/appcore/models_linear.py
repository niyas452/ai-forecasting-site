import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet


def _slice_X_for_ticker(X: pd.DataFrame, ticker: str) -> pd.DataFrame:
    suffix = f"_{ticker}"
    cols = [c for c in X.columns if c.endswith(suffix)]
    return X[cols].copy()


class ElasticNetModel:

    def __init__(self, alpha=0.5, l1_ratio=0.5, random_state=42):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.models: dict[str, object] = {}
        self.tickers: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.tickers = list(y.columns)
        self.models.clear()

        for t in self.tickers:
            Xt = _slice_X_for_ticker(X, t)
            yt = y[t]

            # align + drop NaNs PER ticker
            df = Xt.join(yt.rename("y"), how="inner")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if df.empty or len(df) < 30:
                continue

            Xtt = df.drop(columns=["y"]).values
            ytt = df["y"].values

            model = make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    random_state=self.random_state,
                    max_iter=20000,
                ),
            )
            model.fit(Xtt, ytt)
            self.models[t] = model

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.tickers is None:
            raise RuntimeError("ElasticNetModel must be fit before predict().")

        out = {}
        for t in self.tickers:
            mdl = self.models.get(t)
            Xt = _slice_X_for_ticker(X, t)

            if mdl is None or Xt.empty:
                out[t] = np.nan
                continue

            # use last row that has NO NaNs
            Xt2 = Xt.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if Xt2.empty:
                out[t] = np.nan
                continue

            x_last = Xt2.values[-1].reshape(1, -1)
            out[t] = float(mdl.predict(x_last)[0])

        return pd.Series(out)



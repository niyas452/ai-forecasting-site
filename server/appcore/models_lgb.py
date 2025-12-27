import numpy as np
import pandas as pd
import lightgbm as lgb


def _slice_X_for_ticker(X: pd.DataFrame, ticker: str) -> pd.DataFrame:
    suffix = f"_{ticker}"
    cols = [c for c in X.columns if c.endswith(suffix)]
    return X[cols].copy()


class LGBMQuantile:
    
    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.base_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.models: dict[str, dict[str, lgb.LGBMRegressor]] = {}
        self.tickers: list[str] | None = None

    @staticmethod
    def _clean_join(Xt: pd.DataFrame, yt: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Join X and y and drop invalid rows per ticker.
        LightGBM can handle NaN, but we still remove infs and fully-missing rows.
        """
        df = Xt.join(yt.rename("y"), how="inner")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
        Xt2 = df.drop(columns=["y"])
        yt2 = df["y"]
        return Xt2, yt2

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        common_idx = X.index.intersection(y.index)
        Xc = X.loc[common_idx]
        yc = y.loc[common_idx]

        self.tickers = list(yc.columns)
        self.models.clear()

        for t in self.tickers:
            Xt = _slice_X_for_ticker(Xc, t)
            yt = yc[t]

            if Xt.empty:
                continue

            Xt, yt = self._clean_join(Xt, yt)

            # Minimum samples to train something meaningful
            if len(Xt) < 60:
                continue

            m10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **self.base_params)
            m50 = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **self.base_params)
            m90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **self.base_params)

            m10.fit(Xt, yt)
            m50.fit(Xt, yt)
            m90.fit(Xt, yt)

            self.models[t] = {"p10": m10, "p50": m50, "p90": m90}

        return self

    def predict_quantiles(self, X: pd.DataFrame):
        if self.tickers is None:
            raise RuntimeError("LGBMQuantile must be fit before predict().")

        p10, p50, p90 = {}, {}, {}

        for t in self.tickers:
            if t not in self.models:
                p10[t] = np.nan
                p50[t] = np.nan
                p90[t] = np.nan
                continue

            Xt = _slice_X_for_ticker(X, t)
            if Xt.empty:
                p10[t] = np.nan
                p50[t] = np.nan
                p90[t] = np.nan
                continue

            Xt = Xt.replace([np.inf, -np.inf], np.nan)

            # Use last available row
            x_last = Xt.iloc[[-1]]

            p10[t] = float(self.models[t]["p10"].predict(x_last)[0])
            p50[t] = float(self.models[t]["p50"].predict(x_last)[0])
            p90[t] = float(self.models[t]["p90"].predict(x_last)[0])

        return pd.Series(p10), pd.Series(p50), pd.Series(p90)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        _, p50, _ = self.predict_quantiles(X)
        return p50
)

import numpy as np
import pandas as pd


class PerformanceWeightedEnsemble:
   

    def __init__(self):
        self.models: dict[str, object] = {}
        self.weights: dict[str, float] = {}

    def set_models(self, **models):
        # Keep only non-None models
        self.models = {k: v for k, v in models.items() if v is not None}
        n = len(self.models)
        if n == 0:
            raise RuntimeError("No models provided to ensemble.")
        self.weights = {name: 1.0 / n for name in self.models}
        return self

    def predict(self, X) -> pd.Series:
        preds = []
        for name, mdl in self.models.items():
            s = mdl.predict(X)
            s.name = name
            preds.append(s)

        P = pd.concat(preds, axis=1)  # rows=tickers, cols=models
        w = pd.Series(self.weights).reindex(P.columns).fillna(0.0)

        # weighted mean; ignore missing model outputs per ticker
        num = (P.mul(w, axis=1)).sum(axis=1, skipna=True)
        den = (~P.isna()).mul(w, axis=1).sum(axis=1).replace(0, np.nan)
        return (num / den)


import pandas as pd


class PerformanceWeightedEnsemble:
    """
    Simple equal-weight ensemble (for now):
    - set_models(ElasticNet=enet, LightGBM=lgb, LSTM=lstm)
    - predict(X) -> Series indexed by ticker
    """

    def __init__(self, lamb: float = 3.0):
        self.models: dict[str, object] = {}
        self.weights: dict[str, float] | None = None
        self.lamb = lamb

    def set_models(self, **models):
        self.models = models
        n = len(models)
        # equal weights for now
        self.weights = {name: 1.0 / n for name in models}
        return self

    def predict(self, X) -> pd.Series:
        if not self.models:
            raise RuntimeError("No models set in ensemble.")

        preds = []
        for name, mdl in self.models.items():
            s = mdl.predict(X)  # Series indexed by ticker
            s.name = name
            preds.append(s)

        P = pd.concat(preds, axis=1)  # columns = model names, index = tickers

        # weighted mean across models
        if self.weights is None:
            w = {c: 1.0 / P.shape[1] for c in P.columns}
        else:
            w = self.weights

        # ensure alignment
        w_vec = pd.Series(w).reindex(P.columns).fillna(0.0)
        # dot product along columns → Series indexed by ticker
        return (P * w_vec).sum(axis=1)

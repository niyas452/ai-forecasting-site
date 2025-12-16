import numpy as np
import pandas as pd


class ConformalBands:
    """Simple absolute-residual conformal wrapper.
    Fit on validation residuals; produce symmetric P10/P90 around median.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_: float | None = None

    def fit(self, y_true: pd.Series, y_pred: pd.Series):
        # absolute residuals on validation set
        resid = (y_true - y_pred).abs().dropna()
        if resid.empty:
            self.q_ = None
            return self
        self.q_ = float(resid.quantile(1 - self.alpha))
        return self

    def predict_interval(self, y_med: pd.Series):
        # If not fitted or no residuals, return NaN bands
        if self.q_ is None:
            nan = np.nan
            return y_med * 0 + nan, y_med, y_med * 0 + nan
        lo = y_med - self.q_
        hi = y_med + self.q_
        return lo, y_med, hi

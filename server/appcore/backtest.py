import numpy as np
import pandas as pd
from typing import Callable, Dict, Tuple

from appcore.utils import to_monthly
from appcore.features import make_monthly_features, targets


# =========================================================
# Error metrics
# =========================================================

def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    seasonal_period: int = 12,
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    """
    y_true, y_pred = y_true.align(y_pred, join="inner")

    if y_true.empty:
        return float(np.inf)

    mae = (y_true - y_pred).abs().mean()
    denom = (y_true - y_true.shift(seasonal_period)).abs().dropna().mean()

    if denom is None or denom == 0 or not np.isfinite(denom):
        return float(np.inf)

    return float(mae / denom)


def directional_accuracy(
    preds_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute directional accuracy per ticker.
    """
    preds_df, actuals_df = preds_df.align(actuals_df, join="inner", axis=0)
    preds_df, actuals_df = preds_df.align(actuals_df, join="inner", axis=1)

    da = {}
    for t in preds_df.columns:
        correct = (
            ((preds_df[t] > 0) & (actuals_df[t] > 0)) |
            ((preds_df[t] < 0) & (actuals_df[t] < 0))
        )
        da[t] = float(correct.mean())

    return pd.Series(da, name="DA")


# =========================================================
# Walk-forward backtesting
# =========================================================

def walk_forward(
    prices: pd.DataFrame,
    make_models_fn: Callable[[], Dict[str, object]],
    horizon: str,
    start_year: int = 2012,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward validation.
    """
    m = to_monthly(prices)
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = m.index[m.index.year >= start_year]
    preds, actuals = [], []

    for i in range(len(dates) - 1):
        dt = dates[i]
        next_dt = dates[i + 1]

        hist = m.loc[:dt]
        if len(hist) < 60:
            continue

        X = make_monthly_features(hist)
        y = targets(hist, horizon)
        if X.empty or y.empty:
            continue

        models = make_models_fn()
        for mdl in models.values():
            mdl.fit(X, y)

        X_next = make_monthly_features(m.loc[:next_dt])
        y_next = targets(m.loc[:next_dt], horizon)
        if X_next.empty or y_next.empty:
            continue

        y_act = y_next.iloc[-1]

        preds_step = []
        for mdl in models.values():
            preds_step.append(mdl.predict(X_next))

        P = pd.concat(preds_step, axis=1)
        ens = P.mean(axis=1)

        preds.append(ens.rename(next_dt))
        actuals.append(y_act.rename(next_dt))

    if not preds:
        return pd.DataFrame(), pd.DataFrame()

    preds_df = pd.DataFrame(preds)
    actuals_df = pd.DataFrame(actuals)

    preds_df, actuals_df = preds_df.align(actuals_df, join="inner", axis=0)
    preds_df, actuals_df = preds_df.align(actuals_df, join="inner", axis=1)

    return preds_df, actuals_df


# =========================================================
# Metrics computation
# =========================================================

def compute_backtest_metrics(
    preds_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MASE, and Directional Accuracy.
    """
    rows = []

    for t in preds_df.columns:
        y_true = actuals_df[t]
        y_pred = preds_df[t]

        err = y_true - y_pred
        mae = float(err.abs().mean())
        rmse = float(np.sqrt((err ** 2).mean()))
        mase_val = mase(y_true, y_pred, seasonal_period)

        rows.append({
            "ticker": t,
            "MAE": mae,
            "RMSE": rmse,
            "MASE": mase_val,
        })

    metrics = pd.DataFrame(rows).set_index("ticker")

    da = directional_accuracy(preds_df, actuals_df)
    metrics["DA"] = da

    metrics.loc["ALL"] = {
        "MAE": metrics["MAE"].mean(),
        "RMSE": metrics["RMSE"].mean(),
        "MASE": metrics["MASE"].mean(),
        "DA": metrics["DA"].mean(),
    }

    return metrics


# =========================================================
# Run as script
# =========================================================

if __name__ == "__main__":
    from appcore.data import load_prices
    from appcore.models_linear import ElasticNetModel
    from appcore.models_lgb import LGBMQuantile
    from appcore.models_lstm import LSTMModel

    def make_models() -> Dict[str, object]:
        return {
            "ElasticNet": ElasticNetModel(),
            "LightGBM": LGBMQuantile(quantile=0.5),
            "LSTM": LSTMModel(epochs=10),
        }

    tickers = ["AAPL", "MSFT", "SPY"]
    print(f"Downloading prices for {tickers} ...")
    prices = load_prices(tickers)

    print("Running walk-forward backtest (6m horizon)...")
    preds, actuals = walk_forward(prices, make_models, horizon="6m")

    if preds.empty:
        print("Backtest produced no data.")
    else:
        print("Computing metrics...")
        metrics = compute_backtest_metrics(preds, actuals)
        print(metrics)

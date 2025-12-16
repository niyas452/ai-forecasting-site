import os
import sys
import json
import re
import math

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from appcore.data import load_prices
from appcore.features import make_monthly_features, targets
from appcore.models_linear import ElasticNetModel
from appcore.models_lgb import LGBMQuantile
from appcore.models_lstm import LSTMModel
from appcore.ensemble import PerformanceWeightedEnsemble
from appcore.conformal import ConformalBands
from appcore.risk import ledoit_wolf_cov
from appcore.optimize import max_sharpe
from appcore.utils import to_monthly

# ---------- path bootstrap ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ---------- helpers ----------

def safe_float(x):
    """Convert to a normal finite float or return None."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_tickers(raw: str) -> list[str]:
    """
    Accepts either 'AAPL,MSFT,SPY' or '["AAPL","MSFT","SPY"]'
    and returns ['AAPL','MSFT','SPY'].
    Also strips stray quotes/brackets/spaces.
    """
    cleaned = raw.strip()
    # Try JSON array first
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            cands = obj
        else:
            cands = [cleaned]
    except Exception:
        # Fallback to CSV / whitespace split
        cands = re.split(r"[,\s]+", cleaned)

    out: list[str] = []
    for t in cands:
        t = str(t).strip().upper()
        t = t.strip(' "\'[]')
        if t:
            out.append(t)
    return out


# ---------- app + schemas ----------

app = FastAPI(title="AI Ensemble Forecasting API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


class ForecastResponse(BaseModel):
    ticker: str
    horizon: str
    p10: float | None = None
    p50: float
    p90: float | None = None
    spot: float | None = None          # current price
    price_p50: float | None = None     # expected price at horizon (from p50)


class OptimizeRequest(BaseModel):
    horizon: str = "6m"
    tickers: list[str]


# optional nice root
@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/forecast", "/optimize", "/docs"]}


# ---------- endpoints ----------

@app.get("/forecast")
def forecast(
    tickers: str = Query(..., description="Comma-separated tickers or JSON list"),
    horizon: str = Query("6m", enum=["6m", "12m"]),
):
    # parse input tickers
    tickers_list = parse_tickers(tickers)

    # download prices
    prices = load_prices(tickers_list)
    if prices is None or prices.empty:
        raise HTTPException(
            status_code=502,
            detail="Could not download any price data from Yahoo Finance. "
                   "Try again or change tickers.",
        )

    # build features/targets
    try:
        X = make_monthly_features(prices)
        y = targets(prices, horizon)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Data error while building features: {e}",
        )

        # ----- MODELS (deterministic: ElasticNet only for stable outputs) -----
    enet = ElasticNetModel().fit(X, y)

    # Directly use ElasticNet predictions as our median forecast (p50)
    p50_series = enet.predict(X)  # Series indexed by ticker


    # --- Conformal interval using last ~24 months residuals (safe) ---
    try:
        # Only use dates that exist in BOTH X and y
        val_idx = X.index.intersection(y.index)
        if val_idx.empty:
            raise ValueError("No overlapping dates between X and y for conformal.")

        # Take up to the last 24 dates
        if len(val_idx) > 24:
            val_idx = val_idx[-24:]

        # True values stacked as (date, ticker)
        y_val = y.loc[val_idx].stack()

        # Use the SAME feature matrix the model saw at training
        y_hat_matrix = enet.model.predict(X.loc[val_idx])  # (n_samples, n_tickers)

        # Wrap into DataFrame with same columns (tickers)
        y_hat_df = pd.DataFrame(y_hat_matrix, index=val_idx, columns=y.columns)

        # Stack to align with y_val
        y_hat_val = y_hat_df.stack()

        conformal = ConformalBands(alpha=0.1).fit(y_val, y_hat_val)
        p10_series, _, p90_series = conformal.predict_interval(p50_series)
    except Exception as e:
        print(f"[WARN] conformal interval failed: {e}")
        # fallback: treat as "no interval"
        p10_series = pd.Series(index=p50_series.index, dtype=float)
        p90_series = pd.Series(index=p50_series.index, dtype=float)

    # last available prices per ticker
    last_prices = prices.iloc[-1]

    out: list[ForecastResponse] = []
    for t in tickers_list:
        # current price
        spot_val = (
            safe_float(last_prices.get(t)) if t in last_prices.index else None
        )

        # median log-return
        p50_val = safe_float(p50_series.get(t))

        # conformal bounds (may be None)
        p10_val = None
        p90_val = None
        if isinstance(p10_series, pd.Series) and t in p10_series.index:
            p10_val = safe_float(p10_series[t])
        if isinstance(p90_series, pd.Series) and t in p90_series.index:
            p90_val = safe_float(p90_series[t])

        # expected future price from log-return: S_T = S_0 * exp(p50)
        price_p50_val = None
        if spot_val is not None and p50_val is not None:
            price_p50_val = safe_float(spot_val * math.exp(p50_val))

        out.append(
            ForecastResponse(
                ticker=t,
                horizon=horizon,
                p10=p10_val,
                p50=p50_val if p50_val is not None else 0.0,
                p90=p90_val,
                spot=spot_val,
                price_p50=price_p50_val,
            )
        )

    return {"forecasts": [o.dict() for o in out]}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    # normalize tickers: ensure upper-case symbols
    tickers_list = [t.upper() for t in req.tickers]

    # download prices
    prices = load_prices(tickers_list)
    if prices is None or prices.empty:
        raise HTTPException(
            status_code=502,
            detail="Could not download any price data from Yahoo Finance for optimization.",
        )

    # features/targets
    try:
        X = make_monthly_features(prices)
        y = targets(prices, req.horizon)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"Data error while building features: {e}",
        )

        # ----- MODELS (deterministic ElasticNet only) -----
    enet = ElasticNetModel().fit(X, y)

    # Use ElasticNet predictions as expected log-returns μ
    mu = enet.predict(X)  # expected log-return (median) per ticker


    # risk: Ledoit–Wolf covariance on monthly returns
    m = to_monthly(prices)
    rets = m.pct_change().dropna()
    cov = ledoit_wolf_cov(rets)

    # align and optimize
    common = mu.index.intersection(cov.index)
    if common.empty:
        raise HTTPException(
            status_code=500,
            detail="No overlapping tickers between forecast and covariance matrix.",
        )

    w = max_sharpe(mu.loc[common], cov.loc[common, common])

    return {"weights": w.to_dict(), "mu": mu.to_dict()}

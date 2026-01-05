# app.py
# Main entry point for the forecasting API.
# Handles data fetching, model training on-the-fly, and portfolio optimization.

import os
import sys
import json
import re
import math
import time
from typing import Optional, Dict, Tuple, List, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add appcore to path to allow direct imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from appcore.data import load_prices
from appcore.features import make_monthly_features, targets
from appcore.models_linear import ElasticNetModel
from appcore.models_lgb import LGBMQuantile
from appcore.models_lstm import LSTMModel
from appcore.ensemble import PerformanceWeightedEnsemble
from appcore.risk import ledoit_wolf_cov
from appcore.optimize import max_sharpe_long_only
from appcore.utils import to_monthly


# In-memory cache for stock prices (15 min TTL).
# Prevents spamming Yahoo Finance for repeated requests.
CACHE_TTL_SECONDS = 900
_PRICE_CACHE: Dict[Tuple[str, ...], tuple[float, pd.DataFrame]] = {}


def load_prices_cached(tickers: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    key = tuple(sorted([t.upper() for t in tickers]))
    now = time.time()
    if key in _PRICE_CACHE:
        ts, df = _PRICE_CACHE[key]
        if (now - ts) < CACHE_TTL_SECONDS and df is not None and not df.empty:
            return df
    df = load_prices(list(key), start=start)
    _PRICE_CACHE[key] = (now, df)
    return df


# Helper functions for data parsing and safety checks
def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def last_price_for_ticker(prices: pd.DataFrame, t: str) -> Optional[float]:
    # Returns last non-NaN price for a specific ticker.
    # Essential because tickers may have different trading days/delisting dates.
    if prices is None or prices.empty or t not in prices.columns:
        return None
    s = prices[t].replace([np.inf, -np.inf], np.nan).dropna()
    return safe_float(s.iloc[-1]) if not s.empty else None


def parse_tickers(raw: str) -> list[str]:
    
    cleaned = raw.strip()
    try:
        obj = json.loads(cleaned)
        cands = obj if isinstance(obj, list) else [cleaned]
    except Exception:
        cands = re.split(r"[,\s]+", cleaned)

    out: list[str] = []
    for t in cands:
        t = str(t).strip().upper().strip(' "\'[]')
        if t:
            out.append(t)

    # dedupe preserve order
    seen = set()
    deduped = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def weights_to_pct_100(w: pd.Series, decimals: int = 1) -> pd.Series:
    # Forces weights to sum to exactly 100.0% by adjusting the largest remainder.
    # avoids "99.9%" display issues.
    if w is None or w.empty:
        return pd.Series(dtype=float)

    pct = (w * 100.0).copy()
    factor = 10**decimals
    pct_rounded = (pct * factor).round() / factor
    diff = 100.0 - float(pct_rounded.sum())
    last_key = pct_rounded.index[-1]
    pct_rounded.loc[last_key] = float((pct_rounded.loc[last_key] + diff) * factor) / factor
    return pct_rounded


def make_chart_series(
    prices: pd.DataFrame,
    ticker: str,
    horizon: str,
    forecast_price: Optional[float],
) -> list[dict]:
    # Prepares data for 5-point path charts (History + Current + Forecast).
    # 6m horizon = 18m history; 12m horizon = 36m history.
    m = to_monthly(prices)
    if ticker not in m.columns:
        return []

    if horizon == "6m":
        hist_months = 18
        future_months = 6
    else:
        hist_months = 36
        future_months = 12

    s = m[ticker].dropna()
    if s.empty:
        return []

    s_hist = s.iloc[-(hist_months + 1):]

    pts: list[dict] = []
    for dt, val in s_hist.items():
        pts.append({"date": dt.strftime("%Y-%m"), "price": float(val), "kind": "actual"})

    if forecast_price is not None and len(pts) > 0:
        last_dt = s_hist.index[-1]
        future_dt = last_dt + pd.DateOffset(months=future_months)
        pts.append({"date": future_dt.strftime("%Y-%m"), "price": float(forecast_price), "kind": "forecast"})

    return pts


# Pydantic models for request/response validation
class ChartPoint(BaseModel):
    date: str
    price: float
    kind: Literal["actual", "forecast"]


class ForecastResponse(BaseModel):
    ticker: str
    horizon: str
    p10: Optional[float] = None
    p50: float
    p90: Optional[float] = None
    spot: Optional[float] = None
    price_p50: Optional[float] = None
    chart_series: List[ChartPoint] = []


class OptimizeRequest(BaseModel):
    horizon: str = "6m"
    tickers: list[str]
    max_weight: float = 0.60


# FastAPI Application setup
app = FastAPI(title="AI Ensemble Forecasting API", version="0.6.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/forecast", "/optimize", "/docs"]}


# Model training and inference logic
def fit_models_for_forecast(X: pd.DataFrame, y: pd.DataFrame):
    enet = ElasticNetModel().fit(X, y)

    qgbm = None
    try:
        qgbm = LGBMQuantile().fit(X, y)
    except Exception as e:
        print(f"[WARN] Quantile model failed: {e}")

    lstm = None
    try:
        lstm = LSTMModel(seq_len=24, epochs=20).fit(X, y)
    except Exception as e:
        print(f"[WARN] LSTM failed: {e}")

    ens = PerformanceWeightedEnsemble().set_models(
        ElasticNet=enet,
        QuantileGBM=qgbm,
        LSTM=lstm,
    )
    return enet, qgbm, ens


# API Endpoints
@app.get("/forecast")
def forecast(
    tickers: str = Query(..., description="Comma-separated tickers or JSON list"),
    horizon: str = Query("6m", enum=["6m", "12m"]),
):
    tickers_list = parse_tickers(tickers)

    prices = load_prices_cached(tickers_list)
    if prices is None or prices.empty:
        raise HTTPException(502, "Could not download any price data from Yahoo Finance.")

    try:
        X = make_monthly_features(prices)
        y = targets(prices, horizon)
    except Exception as e:
        raise HTTPException(502, f"Data error while building features: {e}")

    enet, qgbm, ens = fit_models_for_forecast(X, y)

    # Ensemble p50 (expected log-return over horizon)
    p50_series = ens.predict(X).reindex(tickers_list)

    # Intervals: prefer quantiles if available, else N/A
    p10_series = pd.Series(index=p50_series.index, dtype=float)
    p90_series = pd.Series(index=p50_series.index, dtype=float)

    if qgbm is not None:
        try:
            p10_q, p50_q, p90_q = qgbm.predict_quantiles(X)
            p10_series = p10_q.reindex(p50_series.index)
            p90_series = p90_q.reindex(p50_series.index)
            p50_series = p50_series.fillna(p50_q.reindex(p50_series.index))
        except Exception as e:
            print(f"[WARN] Quantile prediction failed: {e}")

    out: list[dict] = []
    for t in tickers_list:
        # Get spot price safely
        spot_val = last_price_for_ticker(prices, t)

        p50_val = safe_float(p50_series.get(t))
        p10_val = safe_float(p10_series.get(t))
        p90_val = safe_float(p90_series.get(t))

        # expected future price from log-return: S_T = S_0 * exp(p50)
        price_p50_val = None
        if spot_val is not None and p50_val is not None:
            price_p50_val = safe_float(spot_val * math.exp(p50_val))

        chart_series = make_chart_series(prices, t, horizon, price_p50_val)

        item = ForecastResponse(
            ticker=t,
            horizon=horizon,
            p10=p10_val,
            p50=p50_val if p50_val is not None else 0.0,
            p90=p90_val,
            spot=spot_val,
            price_p50=price_p50_val,
            chart_series=chart_series,
        ).dict()

        out.append(item)

    return {"forecasts": out}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    tickers_list = [t.upper() for t in req.tickers]

    prices = load_prices_cached(tickers_list)
    if prices is None or prices.empty:
        raise HTTPException(502, "Could not download any price data from Yahoo Finance for optimization.")

    try:
        X = make_monthly_features(prices)
        y = targets(prices, req.horizon)
    except Exception as e:
        raise HTTPException(502, f"Data error while building features: {e}")

    # FAST μ for optimization
    enet = ElasticNetModel().fit(X, y)
    mu_log = enet.predict(X).reindex(tickers_list)

    # Ledoit–Wolf covariance on historical LOG returns
    m = to_monthly(prices)
    rets_log = np.log(m / m.shift(1)).dropna(how="any")
    cov = ledoit_wolf_cov(rets_log)

    common = mu_log.index.intersection(cov.index)
    if common.empty:
        raise HTTPException(500, "No overlapping tickers between forecast and covariance matrix.")

    mu2 = mu_log.loc[common]
    cov2 = cov.loc[common, common]

    w = max_sharpe_long_only(mu2, cov2, risk_free=0.0, max_w=float(req.max_weight))
    w_pct = weights_to_pct_100(w, decimals=1)

    return {
        "weights": w.to_dict(),
        "weights_pct": w_pct.to_dict(),
        "mu": mu_log.to_dict(),
        "max_weight": float(req.max_weight),
    }

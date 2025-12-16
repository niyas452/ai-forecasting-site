import os
import time
import pandas as pd
import yfinance as yf


def _extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Handle both old and new yfinance formats and return a 1D Close price Series.
    """
    if df is None or df.empty:
        return None

    # New yfinance: MultiIndex columns (e.g. ('Price','Close','SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        # Try to select level that looks like 'Close'
        try:
            # level names often like ('Price', 'Close', 'SPY')
            if "Close" in df.columns.get_level_values(1):
                close = df.xs("Close", axis=1, level=1)
            elif "Adj Close" in df.columns.get_level_values(1):
                close = df.xs("Adj Close", axis=1, level=1)
            else:
                # fall back to first level=1 column
                close = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
        except Exception:
            return None

        # If single ticker, close may be a DataFrame with one column
        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            s = close.iloc[:, 0]
        else:
            s = close

        s = s.astype(float)
        s.name = ticker
        return s.dropna()

    # Old yfinance: flat columns like 'Adj Close', 'Close'
    for col in ["Adj Close", "Close", "close", "adj_close"]:
        if col in df.columns:
            s = df[col].astype(float)
            s.name = ticker
            return s.dropna()

    return None


def _download_one(ticker: str, start: str = "2005-01-01") -> pd.Series | None:
    """Download a single ticker robustly; return its Close as a Series or None."""
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                auto_adjust=True,   # now defaults to True in new yfinance
                progress=False,
                actions=False,
                threads=False,
                interval="1d",
            )
            s = _extract_close_series(df, ticker)
            if s is not None and not s.empty:
                return s
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


def load_prices(tickers, start: str = "2005-01-01") -> pd.DataFrame:
    """Download multiple tickers one-by-one with robust parsing."""
    if isinstance(tickers, str):
        tickers = [tickers]

    series = []
    for t in tickers:
        s = _download_one(t.strip().upper(), start=start)
        if s is not None and not s.empty:
            series.append(s)

    if not series:
        # return an empty DataFrame; caller / API will handle with HTTPException
        return pd.DataFrame()

    out = pd.concat(series, axis=1).ffill().dropna(how="all")
    # Drop tickers that ended up totally NA
    out = out.dropna(axis=1, how="all")
    return out

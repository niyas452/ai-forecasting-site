import time
import pandas as pd
import yfinance as yf


def _extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Handle both old and new yfinance formats and return a 1D Close price Series.
    """
    if df is None or df.empty:
        return None

    # New yfinance: MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # We need to find which level contains 'Close' or 'Adj Close'
        # in yfinance 1.0+, Level 0 is often 'Price' (Close, Open, etc)
        # in some older versions/calls, it might be reversed or different.
        
        target_col = "Close"
        
        # Check if 'Price' level is level 0
        level0_vals = df.columns.get_level_values(0)
        
        # If 'Close' is in level 0 (e.g. ('Close', 'AAPL'))
        if "Close" in level0_vals:
            close = df.xs("Close", axis=1, level=0)
        elif "Adj Close" in level0_vals:
            close = df.xs("Adj Close", axis=1, level=0)
        # Fallback: check level 1
        elif df.columns.nlevels > 1 and "Close" in df.columns.get_level_values(1):
            close = df.xs("Close", axis=1, level=1)
        elif df.columns.nlevels > 1 and "Adj Close" in df.columns.get_level_values(1):
            close = df.xs("Adj Close", axis=1, level=1)
        else:
             # Just take the first column's group? Risky.
             # Let's try to grab the first column of the df if nothing matches (fallback)
             # But we need a Series.
             # Let's just return None if we can't find Close.
             return None

        if isinstance(close, pd.DataFrame):
            if close.shape[1] == 0:
                return None
            # if multiple columns exist, try ticker name match; else take first
            if ticker in close.columns:
                s = close[ticker]
            else:
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
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                actions=False,
                threads=False,
                interval="1d",
            )
            s = _extract_close_series(df, ticker)
            if s is not None and not s.empty:
                # normalize index + sort + drop duplicates
                s.index = pd.to_datetime(s.index)
                s = s[~s.index.duplicated(keep="last")].sort_index()
                return s
            time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


def load_prices(tickers, start: str = "2005-01-01") -> pd.DataFrame:
    
    if isinstance(tickers, str):
        tickers = [tickers]

    series = []
    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue
        s = _download_one(t, start=start)
        if s is not None and not s.empty:
            series.append(s)

    if not series:
        return pd.DataFrame()

    out = pd.concat(series, axis=1)

    # normalize index, sort
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # keep only columns that actually have data
    out = out.dropna(axis=1, how="all")

    # keep rows if at least one ticker has data
    out = out.dropna(how="all")

    return out

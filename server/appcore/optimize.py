# appcore/optimize.py
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def max_sharpe_unconstrained(mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
   
    tickers = list(mu.index)
    mu_vec = mu.values.reshape(-1, 1).astype(float)
    inv_cov = np.linalg.pinv(cov.values.astype(float))
    ones = np.ones((len(tickers), 1))
    excess = mu_vec - float(risk_free)
    w = inv_cov @ excess
    denom = float(ones.T @ w)
    if abs(denom) < 1e-12:
        return pd.Series(1.0 / len(tickers), index=tickers)
    w = (w / denom).flatten()
    return pd.Series(w, index=tickers)


def _normalize_simplex_with_cap(w: pd.Series, max_w: float = 1.0) -> pd.Series:
    """
    Long-only projection-ish heuristic:
      clip [0, max_w], then renormalize to sum=1.
    """
    w2 = w.copy().astype(float)
    w2 = w2.clip(lower=0.0, upper=float(max_w))
    s = float(w2.sum())
    if s <= 1e-12:
        return pd.Series(1.0 / len(w2), index=w2.index)
    w2 = w2 / s
   
    w2 = w2.clip(lower=0.0, upper=float(max_w))
    s2 = float(w2.sum())
    return (w2 / s2) if s2 > 1e-12 else pd.Series(1.0 / len(w2), index=w2.index)


def max_sharpe_long_only(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free: float = 0.0,
    max_w: float = 1.0,
) -> pd.Series:
    """
    Maximize Sharpe ratio subject to:
      sum(w)=1
      0 <= w_i <= max_w
    Uses SLSQP if SciPy is installed; otherwise falls back to a heuristic.
    """
    tickers = list(mu.index)
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)

    mu_vec = (mu.values.astype(float) - float(risk_free))
    cov_mat = cov.values.astype(float)
    n = len(tickers)

   
    if _HAS_SCIPY:
        def neg_sharpe(w):
            w = np.asarray(w, dtype=float)
            ret = float(w @ mu_vec)
            vol2 = float(w @ cov_mat @ w)
            if vol2 <= 1e-12:
                return 1e6
            return -(ret / np.sqrt(vol2))

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, float(max_w))] * n
        x0 = np.ones(n) / n

        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if (not res.success) or np.any(~np.isfinite(res.x)):
            # fallback: equal weight
            return pd.Series(1.0 / n, index=tickers)

        w = pd.Series(res.x, index=tickers).astype(float)
        # numerical cleanup
        w[w < 0] = 0.0
        s = float(w.sum())
        return (w / s) if s > 1e-12 else pd.Series(1.0 / n, index=tickers)

    # No SciPy fallback:
    # 1) compute unconstrained tangency
    w0 = max_sharpe_unconstrained(mu, cov, risk_free=risk_free)
    # 2) enforce long-only + cap + renormalize
    return _normalize_simplex_with_cap(w0, max_w=max_w)

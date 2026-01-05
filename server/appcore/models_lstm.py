# models_lstm.py
# PyTorch LSTM implementation for sequence-based forecasting.
# Enforces determinism for reproducible research results.

import os
import random
import zlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def _slice_X_for_ticker(X: pd.DataFrame, ticker: str) -> pd.DataFrame:
    suffix = f"_{ticker}"
    cols = [c for c in X.columns if c.endswith(suffix)]
    return X[cols].copy()


class _LSTMReg(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class LSTMModel:
    

    def __init__(
        self,
        seq_len: int = 24,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        device=None,
        seed: int = 42,
        hidden: int = 64,
        dropout: float = 0.0,
        deterministic: bool = True,
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = int(seed)
        self.hidden = hidden
        self.dropout = dropout
        self.deterministic = deterministic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models: dict[str, _LSTMReg] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.tickers: list[str] | None = None

    def _set_global_deterministic(self):
        # global deterministic config (once)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # if this errors on your setup, it's okay to continue
            pass

    def _seed_for_ticker(self, ticker: str) -> int:
        # stable hash across runs (unlike Python's hash())
        h = zlib.crc32(ticker.encode("utf-8")) & 0xFFFFFFFF
        return (self.seed + int(h)) % (2**31 - 1)

    def _reseed_all(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.deterministic:
            self._set_global_deterministic()

        self.tickers = list(y.columns)
        self.models.clear()
        self.scalers.clear()

        for t in self.tickers:
            # Seed isolation per ticker prevents cross-contamination of randomness.
            t_seed = self._seed_for_ticker(t)
            self._reseed_all(t_seed)

            Xt = _slice_X_for_ticker(X, t)
            yt = y[t]

            # align + drop NaNs per ticker
            df = Xt.join(yt.rename("y"), how="inner")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if df.empty or len(df) <= self.seq_len + 1:
                continue

            Xt2 = df.drop(columns=["y"])
            yt2 = df["y"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(Xt2.values.astype(np.float32))
            y_arr = yt2.values.astype(np.float32)

            X_seq, y_seq = [], []
            for i in range(self.seq_len, len(X_scaled)):
                X_seq.append(X_scaled[i - self.seq_len : i, :])
                y_seq.append(y_arr[i])

            X_seq = np.asarray(X_seq, dtype=np.float32)
            y_seq = np.asarray(y_seq, dtype=np.float32)

            ds = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))

            # Use a localized generator for data shuffling.
            g = torch.Generator()
            g.manual_seed(t_seed)

            dl = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                generator=g,
                num_workers=0,
                pin_memory=False,
            )

            model = _LSTMReg(
                n_features=X_seq.shape[-1],
                hidden=self.hidden,
                dropout=self.dropout,
            ).to(self.device)

            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(self.epochs):
                for xb, yb in dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()

            self.models[t] = model
            self.scalers[t] = scaler

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.tickers is None:
            raise RuntimeError("LSTMModel must be fit before predict().")

        out: dict[str, float] = {}
        for t in self.tickers:
            model = self.models.get(t)
            scaler = self.scalers.get(t)

            if model is None or scaler is None:
                out[t] = float("nan")
                continue

            Xt = _slice_X_for_ticker(X, t)
            Xt2 = Xt.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if len(Xt2) < self.seq_len:
                out[t] = float("nan")
                continue

            X_scaled = scaler.transform(Xt2.values.astype(np.float32))
            last_seq = X_scaled[-self.seq_len :, :]
            xb = torch.tensor(last_seq[None, :, :], dtype=torch.float32).to(self.device)

            model.eval()
            with torch.no_grad():
                out[t] = float(model(xb).item())

        return pd.Series(out)

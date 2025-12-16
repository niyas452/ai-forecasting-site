import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel:
    """A lightweight LSTM regressor for forecasting."""

    def __init__(self, seq_len=36, epochs=50, lr=1e-3, device=None):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_model(self, input_dim):
        model = nn.Sequential(
            nn.LSTM(input_dim, 64, batch_first=True),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        return model

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_med = y.median(axis=1)
        common = X.index.intersection(y_med.index)
        X_arr = X.loc[common].values.astype(np.float32)
        y_arr = y_med.loc[common].values.astype(np.float32)

        # Reshape for LSTM (samples, seq_len, features)
        X_tensor = torch.tensor(X_arr).unsqueeze(1)
        y_tensor = torch.tensor(y_arr).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_dim = X_tensor.shape[-1]
        self.model = self._build_model(input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred, _ = self.model[0](xb)
                out = self.model[2](torch.relu(pred[:, -1, :]))
                loss = criterion(out.squeeze(), yb.squeeze())
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")

        self.model.eval()
        X_tensor = torch.tensor(X.values.astype(np.float32)).unsqueeze(1).to(self.device)
        with torch.no_grad():
            pred, _ = self.model[0](X_tensor)
            out = self.model[2](torch.relu(pred[:, -1, :]))
            preds = out.cpu().numpy().flatten()

        tickers = [c.split("_")[-1] for c in X.columns]
        last = preds[-1] if len(preds) else np.nan
        return pd.Series({t: float(last) for t in tickers})

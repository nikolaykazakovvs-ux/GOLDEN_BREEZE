# aimodule/training/train_direction_lstm.py

"""
Обучение LSTM-модели направления для XAUUSD.
Сохраняет веса в models/direction_model.pt
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ..models.direction_model import LSTMDirectionModel
from ..config import MODELS_DIR, DIRECTION_MODEL_PATH


class CandleSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 50):
        self.seq_len = seq_len
        self.df = df.reset_index(drop=True)
        self.X, self.y = self._build_samples(self.df)

    def _build_samples(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        closes = df["close"].astype(float).values
        vols = df["volume"].astype(float).values
        deltas = np.diff(closes, prepend=closes[0])

        feats = np.stack([closes, deltas, vols], axis=1)
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for i in range(1, len(feats)):
            end = i + 1
            start = max(0, end - self.seq_len)
            seq = feats[start:end]
            if len(seq) < self.seq_len:
                pad = np.repeat(seq[:1, :], self.seq_len - len(seq), axis=0)
                seq = np.concatenate([pad, seq], axis=0)
            X_list.append(seq.astype(np.float32))

            # Класс по изменению цены на следующем шаге
            diff = closes[end - 1] - closes[end - 2]
            if abs(diff) < 1e-4:
                y_list.append(0)  # FLAT
            elif diff > 0:
                y_list.append(1)  # LONG
            else:
                y_list.append(2)  # SHORT

        return np.array(X_list), np.array(y_list)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train(df: pd.DataFrame, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    dataset = CandleSeqDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMDirectionModel()
    device = torch.device("cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"Epoch {ep}: loss={total/len(loader):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DIRECTION_MODEL_PATH)
    print(f"Saved weights to {DIRECTION_MODEL_PATH}")


if __name__ == "__main__":
    # Пример: загрузка CSV с историей XAUUSD (timestamp,open,high,low,close,volume)
    data_path = Path("data/xauusd_m5.csv")
    if not data_path.exists():
        raise FileNotFoundError("Не найден data/xauusd_m5.csv. Подготовьте исторические данные.")
    df = pd.read_csv(data_path)
    train(df, epochs=10)

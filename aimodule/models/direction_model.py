# aimodule/models/direction_model.py

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..utils import Direction
from ..config import MODELS_DIR, DIRECTION_MODEL_PATH


class LSTMDirectionModel(nn.Module):
    """
    Лёгкая LSTM-модель для предсказания краткосрочного направления.
    Вход: последовательность признаков (close, delta_close, volume)
    Выход: логиты классов [FLAT, LONG, SHORT]
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 32, num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # берём последний таймстеп
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


class DirectionPredictor:
    """
    Обёртка над LSTMDirectionModel с загрузкой весов и предобработкой данных.
    Если весов нет — используется fallback-логика price momentum.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = LSTMDirectionModel()
        self.fallback = True

        # Загрузка весов, если доступны
        try:
            weights_path = Path(DIRECTION_MODEL_PATH)
            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.fallback = False
        except Exception:
            self.fallback = True

    def _build_sequence(self, df: pd.DataFrame, seq_len: int = 50) -> np.ndarray:
        closes = df["close"].astype(float).values
        vols = df["volume"].astype(float).values
        deltas = np.diff(closes, prepend=closes[0])

        feats = np.stack([closes, deltas, vols], axis=1)
        # Берём последние seq_len шагов, при нехватке — паддинг повтором первого элемента
        if len(feats) < seq_len:
            pad = np.repeat(feats[:1, :], seq_len - len(feats), axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        else:
            feats = feats[-seq_len:]
        return feats.astype(np.float32)

    def predict(self, df: pd.DataFrame) -> Tuple[Direction, float]:
        if len(df) < 2:
            return Direction.FLAT, 0.0

        if self.fallback:
            # Momentum fallback
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
            diff = last - prev
            if abs(diff) < 1e-4:
                return Direction.FLAT, 0.1
            conf = float(min(1.0, abs(diff) / (max(prev, 1e-8) * 0.002 + 1e-8)))
            return (Direction.LONG, conf) if diff > 0 else (Direction.SHORT, conf)

        seq = self._build_sequence(df)
        x = torch.from_numpy(seq).unsqueeze(0)  # (1, seq_len, input_size)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        # FLAT=0, LONG=1, SHORT=2
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        if idx == 0:
            return Direction.FLAT, confidence
        elif idx == 1:
            return Direction.LONG, confidence
        else:
            return Direction.SHORT, confidence

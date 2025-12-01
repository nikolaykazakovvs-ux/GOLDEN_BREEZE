# aimodule/models/regime_model.py

import numpy as np
import pandas as pd
from ..utils import MarketRegime
from ..config import REGIME_MODEL_PATH

from pathlib import Path
from typing import Optional
import joblib


class RegimeClusterModel:
    """
    Модель кластеризации рыночных режимов.
    Ожидает обученный кластеризатор (KMeans/GMM), сохранённый через joblib.
    Признаки: разница SMA, ATR, нормализованный диапазон (high-low)/close.
    При отсутствии модели — простая эвристика.
    """

    def __init__(self):
        self.cluster = None
        path = Path(REGIME_MODEL_PATH)
        if path.exists():
            try:
                self.cluster = joblib.load(path)
            except Exception:
                self.cluster = None

    def _features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if df.empty:
            return None
        last = df.iloc[-1]
        sma_fast = float(last.get("sma_fast", np.nan))
        sma_slow = float(last.get("sma_slow", np.nan))
        atr = float(last.get("atr", np.nan))
        close = float(last.get("close", np.nan))
        high = float(last.get("high", np.nan))
        low = float(last.get("low", np.nan))

        if any(np.isnan([sma_fast, sma_slow, atr, close, high, low])):
            return None

        sma_diff = sma_fast - sma_slow
        rng_norm = (high - low) / (close + 1e-8)
        return np.array([[sma_diff, atr, rng_norm]], dtype=np.float32)

    def _map_cluster_to_regime(self, idx: int, vec: np.ndarray) -> MarketRegime:
        # Если нет явной карты, используем эвристику на основании признаков
        sma_diff, atr, rng_norm = float(vec[0, 0]), float(vec[0, 1]), float(vec[0, 2])
        if abs(sma_diff) < 0.1 * (abs(sma_diff) + 1e-6):
            # near zero diff
            if atr > (atr + 1e-6) * 1.5 or rng_norm > 0.01:
                return MarketRegime.VOLATILE
            return MarketRegime.RANGE
        return MarketRegime.TREND_UP if sma_diff > 0 else MarketRegime.TREND_DOWN

    def predict(self, df: pd.DataFrame) -> MarketRegime:
        vec = self._features(df)
        if vec is None:
            return MarketRegime.UNKNOWN

        if self.cluster is None:
            # Эвристический режим при отсутствии обученной модели
            return self._map_cluster_to_regime(0, vec)

        try:
            idx = int(self.cluster.predict(vec)[0])
            return self._map_cluster_to_regime(idx, vec)
        except Exception:
            return self._map_cluster_to_regime(0, vec)

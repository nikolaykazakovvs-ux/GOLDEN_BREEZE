# aimodule/models/regime_ml_model.py

"""
ML-модель для определения режима рынка (Market Regime Detector).
Использует KMeans или GaussianMixture из scikit-learn.
Признаки: доходности, ATR, наклон SMA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..utils import MarketRegime
from ..config import REGIME_MODEL_PATH


class RegimeMLModel:
    """
    ML-модель кластеризации для определения рыночных режимов.
    
    Использует:
    - KMeans или GMM для кластеризации
    - Признаки: returns, atr, sma_slope, volatility
    - StandardScaler для нормализации
    
    Методы:
    - fit(df): обучение на историческом DataFrame
    - predict(df): предсказание режима для последней свечи
    - save(path): сохранение модели
    - load(path): загрузка модели
    """
    
    def __init__(
        self, 
        method: Literal["kmeans", "gmm"] = "kmeans",
        n_clusters: int = 4
    ):
        """
        Args:
            method: 'kmeans' или 'gmm'
            n_clusters: количество кластеров (режимов)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
        if method == "kmeans":
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            self.clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        
        # Mapping кластер -> режим (заполняется после обучения)
        self.cluster_map = {}
        
    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Извлечение признаков для кластеризации.
        
        Признаки:
        - returns: процентная доходность
        - atr: Average True Range
        - sma_slope: наклон скользящей средней
        - volatility: стандартное отклонение доходностей
        """
        if len(df) < 20:
            return None
            
        required_cols = ["close", "sma_fast", "sma_slow", "atr"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Returns
        returns = df["close"].pct_change().fillna(0).values
        
        # ATR
        atr_values = df["atr"].fillna(df["atr"].mean()).values
        
        # SMA slope (угол наклона быстрой SMA)
        sma_fast = df["sma_fast"].fillna(method='ffill').fillna(method='bfill').values
        if len(sma_fast) >= 5:
            sma_slope = np.gradient(sma_fast[-20:])  # последние 20 значений
        else:
            sma_slope = np.zeros(len(sma_fast))
        
        # Volatility (скользящая волатильность)
        volatility = pd.Series(returns).rolling(window=14, min_periods=1).std().fillna(0).values
        
        # Формируем матрицу признаков
        min_len = min(len(returns), len(atr_values), len(sma_slope), len(volatility))
        
        features = np.column_stack([
            returns[-min_len:],
            atr_values[-min_len:],
            sma_slope[-min_len:] if len(sma_slope) >= min_len else np.zeros(min_len),
            volatility[-min_len:]
        ])
        
        return features.astype(np.float32)
    
    def fit(self, df: pd.DataFrame):
        """
        Обучение модели на историческом DataFrame.
        
        Args:
            df: DataFrame с колонками close, sma_fast, sma_slow, atr
        """
        features = self._extract_features(df)
        if features is None or len(features) < self.n_clusters:
            raise ValueError("Недостаточно данных для обучения")
        
        # Нормализация
        features_scaled = self.scaler.fit_transform(features)
        
        # Кластеризация
        if self.method == "kmeans":
            self.clusterer.fit(features_scaled)
        else:
            self.clusterer.fit(features_scaled)
        
        # Построение mapping кластер -> режим
        self._build_cluster_mapping(features, self.clusterer.predict(features_scaled) if self.method == "kmeans" else self.clusterer.predict(features_scaled))
        
    def _build_cluster_mapping(self, features: np.ndarray, labels: np.ndarray):
        """
        Построение отображения кластер -> MarketRegime на основе характеристик кластеров.
        """
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                self.cluster_map[cluster_id] = MarketRegime.UNKNOWN
                continue
            
            cluster_features = features[mask]
            
            # Средние значения признаков в кластере
            mean_returns = np.mean(cluster_features[:, 0])
            mean_atr = np.mean(cluster_features[:, 1])
            mean_slope = np.mean(cluster_features[:, 2])
            mean_vol = np.mean(cluster_features[:, 3])
            
            # Логика определения режима
            if mean_vol > np.percentile(features[:, 3], 75):
                regime = MarketRegime.VOLATILE
            elif abs(mean_returns) < 0.0005 and mean_atr < np.percentile(features[:, 1], 50):
                regime = MarketRegime.RANGE
            elif mean_slope > 0 and mean_returns > 0:
                regime = MarketRegime.TREND_UP
            elif mean_slope < 0 and mean_returns < 0:
                regime = MarketRegime.TREND_DOWN
            else:
                regime = MarketRegime.RANGE
            
            self.cluster_map[cluster_id] = regime
    
    def predict(self, df: pd.DataFrame) -> MarketRegime:
        """
        Предсказание режима для последней свечи.
        
        Args:
            df: DataFrame с историей
            
        Returns:
            MarketRegime
        """
        features = self._extract_features(df)
        if features is None or len(features) == 0:
            return MarketRegime.UNKNOWN
        
        # Берём последние признаки
        last_features = features[-1:, :]
        
        try:
            # Нормализация
            last_scaled = self.scaler.transform(last_features)
            
            # Предсказание кластера
            if self.method == "kmeans":
                cluster_id = int(self.clusterer.predict(last_scaled)[0])
            else:
                cluster_id = int(self.clusterer.predict(last_scaled)[0])
            
            # Mapping в режим
            return self.cluster_map.get(cluster_id, MarketRegime.UNKNOWN)
        
        except Exception:
            return MarketRegime.UNKNOWN
    
    def save(self, path: Optional[str] = None):
        """Сохранение модели в файл через joblib."""
        if path is None:
            path = REGIME_MODEL_PATH
        
        model_data = {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'clusterer': self.clusterer,
            'scaler': self.scaler,
            'cluster_map': self.cluster_map
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> 'RegimeMLModel':
        """Загрузка модели из файла."""
        if path is None:
            path = REGIME_MODEL_PATH
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")
        
        model_data = joblib.load(path)
        
        instance = cls(
            method=model_data['method'],
            n_clusters=model_data['n_clusters']
        )
        instance.clusterer = model_data['clusterer']
        instance.scaler = model_data['scaler']
        instance.cluster_map = model_data['cluster_map']
        
        return instance

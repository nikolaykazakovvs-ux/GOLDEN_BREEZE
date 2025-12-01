# aimodule/data_pipeline/loader.py

"""
Модуль загрузки и подготовки данных для Golden Breeze v3.0.

Включает:
- Загрузку исторических данных из CSV
- Преобразование Candle объектов в DataFrame
- Подготовку тренировочных данных с полным набором признаков
"""

from typing import List
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils import Candle


def load_history_csv(path: str) -> pd.DataFrame:
    """
    Загрузка исторических данных из CSV файла.
    
    Формат файла:
    - timestamp,open,high,low,close,volume
    - timestamp может быть ISO (2024-01-01T00:00:00) или UNIX timestamp
    
    Args:
        path: путь к CSV файлу
        
    Returns:
        DataFrame с колонками: timestamp, open, high, low, close, volume
        
    Raises:
        FileNotFoundError: если файл не найден
        ValueError: если отсутствуют обязательные колонки
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    df = pd.read_csv(path)
    
    # Проверка обязательных колонок
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    
    # Парсинг timestamp
    try:
        # Попытка парсинга как datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        try:
            # Попытка парсинга как UNIX timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        except Exception:
            # Оставляем как есть, если не удалось распарсить
            pass
    
    # Сортировка по времени
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Преобразование к числовым типам
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Удаление строк с NaN
    df = df.dropna()
    
    return df


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление всех необходимых признаков для обучения моделей.
    
    Добавляемые признаки:
    - returns: простая доходность (close.pct_change)
    - log_returns: логарифмическая доходность
    - sma_fast, sma_slow: скользящие средние (10, 50)
    - atr: Average True Range (14 периодов)
    - volatility: скользящее стандартное отклонение доходностей
    - rsi: Relative Strength Index
    - sma_slope: наклон быстрой SMA
    - price_position: позиция цены относительно диапазона (нормализованная)
    
    Args:
        df: DataFrame с OHLCV данными
        
    Returns:
        DataFrame с добавленными признаками
    """
    from .features import add_basic_features
    
    # Базовые индикаторы (sma_fast, sma_slow, atr)
    df = add_basic_features(df)
    
    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # Volatility (скользящее окно)
    df["volatility"] = df["returns"].rolling(window=14, min_periods=1).std()
    
    # SMA slope (наклон быстрой SMA)
    if "sma_fast" in df.columns:
        df["sma_slope"] = df["sma_fast"].diff()
    else:
        df["sma_slope"] = 0.0
    
    # Price position в диапазоне high-low (нормализованная)
    df["price_range"] = df["high"] - df["low"]
    df["price_position"] = (df["close"] - df["low"]) / (df["price_range"] + 1e-8)
    
    # RSI (опционально, если есть в features.py)
    try:
        import ta
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    except Exception:
        df["rsi"] = 50.0  # нейтральное значение
    
    # Заполнение NaN
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    
    return df


def candles_to_dataframe(candles: List[Candle]) -> pd.DataFrame:
    """
    Преобразование входного списка свечей в pandas.DataFrame.
    """
    data = {
        "timestamp": [c.timestamp for c in candles],
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    df = pd.DataFrame(data)
    return df


def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """
    Простая нормализация признаков.
    Потом сюда можно воткнуть более сложный pipeline из FinRL / TensorTrade.
    """
    features = df[["open", "high", "low", "close", "volume"]].astype(float)
    # Минимально-адекватная нормализация
    norm = (features - features.mean()) / (features.std() + 1e-8)
    return norm.values

# aimodule/data_pipeline/features.py

import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовые фичи: SMA и ATR.
    Это временный костяк — потом сюда добавим фичи, согласованные с твоей стратегией.
    """
    df = df.copy()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Адаптивные окна в зависимости от количества данных
    n = len(df)
    sma_fast_window = min(10, max(2, n // 5))
    sma_slow_window = min(50, max(5, n // 2))
    atr_window = min(14, max(2, n - 1))

    df["sma_fast"] = SMAIndicator(close=close, window=sma_fast_window).sma_indicator()
    df["sma_slow"] = SMAIndicator(close=close, window=sma_slow_window).sma_indicator()
    
    if n >= 2:
        df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=atr_window).average_true_range()
    else:
        df["atr"] = 0.0

    return df

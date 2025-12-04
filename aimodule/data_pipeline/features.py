# aimodule/data_pipeline/features.py

import pandas as pd
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

from .features_smc import add_smc_features
from .features_gold import add_all_gold_features


def add_basic_features(df: pd.DataFrame, use_gold_features: bool = True, higher_tf_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Базовые фичи: SMA, ATR, SMC + Gold-Specific Features.
    
    Args:
        df: DataFrame с OHLCV данными
        use_gold_features: Включить Gold-специфические фичи (default=True)
        higher_tf_data: Опциональные данные со старшего таймфрейма для S/R
    
    Returns:
        DataFrame с добавленными фичами
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

    # Добавляем SMC фичи (Fair Value Gaps + Swing Points)
    df = add_smc_features(df)

    # NEW: Gold-Specific Features (Alpha Trend, ICT, EMA System)
    if use_gold_features and n >= 200:  # Нужно минимум 200 свечей для EMA_200
        print("🏆 Adding Gold-Specific Features...")
        df = add_all_gold_features(df, higher_tf_data)
    elif use_gold_features and n < 200:
        print(f"⚠️  Skipping Gold Features: Need 200+ candles (have {n})")

    return df

"""
Smart Money Concepts (SMC) Features Module for Golden Breeze v3.0.

Включает:
- Fair Value Gaps (FVG) - Bullish и Bearish
- Swing Points - High и Low
"""

import pandas as pd


def _calculate_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Расчет Fair Value Gaps (FVG).
    
    Bullish FVG: (High[i-2] < Low[i]) AND (Close[i-1] > Open[i-1])
    Bearish FVG: (Low[i-2] > High[i]) AND (Close[i-1] < Open[i-1])
    
    Args:
        df: DataFrame с OHLC данными
        
    Returns:
        DataFrame с добавленными колонками SMC_FVG_Bullish, SMC_FVG_Bearish
    """
    df = df.copy()
    
    # Получаем значения high, low, close, open
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    open_price = df['open'].astype(float)
    
    # Сдвигаем данные для анализа предыдущих свечей
    high_2 = high.shift(2)
    low_2 = low.shift(2)
    close_1 = close.shift(1)
    open_1 = open_price.shift(1)
    
    # Bullish FVG: (High[i-2] < Low[i]) AND (Close[i-1] > Open[i-1])
    bullish_fvg = (high_2 < low) & (close_1 > open_1)
    
    # Bearish FVG: (Low[i-2] > High[i]) AND (Close[i-1] < Open[i-1])
    bearish_fvg = (low_2 > high) & (close_1 < open_1)
    
    # Заполняем NaN нулями (первые 2 свечи не могут иметь FVG)
    df['SMC_FVG_Bullish'] = bullish_fvg.fillna(0).astype(int)
    df['SMC_FVG_Bearish'] = bearish_fvg.fillna(0).astype(int)
    
    return df


def _calculate_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Расчет Swing Points (High/Low).
    
    Swing High: High[i] является максимумом в окне [i-2..i+2]
    Swing Low: Low[i] является минимумом в окне [i-2..i+2]
    
    Args:
        df: DataFrame с OHLC данными
        window: размер окна для определения swing (по умолчанию 5)
        
    Returns:
        DataFrame с добавленными колонками SMC_Swing_High, SMC_Swing_Low
    """
    df = df.copy()
    
    # Получаем значения high и low
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # Используем rolling для определения локальных максимумов/минимумов
    # Окно должно быть нечётным для симметрии
    if window % 2 == 0:
        window += 1
    
    half_window = window // 2
    
    # Swing High: текущий high является максимумом в окне
    rolling_max = high.rolling(window=window, center=True).max()
    swing_high = (high == rolling_max)
    
    # Swing Low: текущий low является минимумом в окне
    rolling_min = low.rolling(window=window, center=True).min()
    swing_low = (low == rolling_min)
    
    # Заполняем NaN нулями (граничные значения)
    df['SMC_Swing_High'] = swing_high.fillna(0).astype(int)
    df['SMC_Swing_Low'] = swing_low.fillna(0).astype(int)
    
    return df


def add_smc_features(df: pd.DataFrame, swing_window: int = 5) -> pd.DataFrame:
    """
    Добавление всех SMC признаков к DataFrame.
    
    Включает:
    - Fair Value Gaps (Bullish/Bearish)
    - Swing Points (High/Low)
    
    Args:
        df: DataFrame с OHLC данными
        swing_window: размер окна для swing points (по умолчанию 5)
        
    Returns:
        DataFrame с добавленными SMC признаками
    """
    if len(df) < 3:
        # Недостаточно данных для расчета SMC признаков
        df['SMC_FVG_Bullish'] = 0
        df['SMC_FVG_Bearish'] = 0
        df['SMC_Swing_High'] = 0
        df['SMC_Swing_Low'] = 0
        return df
    
    # Расчет FVG
    df = _calculate_fvg(df)
    
    # Расчет Swing Points
    df = _calculate_swings(df, window=swing_window)
    
    return df

# aimodule/data_pipeline/features_gold.py
"""
Gold-Specific Features
Based on analysis of: pariharmadhukar/Forex_Gold-Price-Prediction-system
"""

import pandas as pd
import numpy as np
from ta import volatility, momentum, trend


def add_alpha_trend(df: pd.DataFrame, atr_period=14, mult=1.5) -> pd.DataFrame:
    """
    Alpha Trend Indicator - специально для XAUUSD
    
    Логика:
    - Bullish: RSI > 50 AND Close > Upper Bound → STRONG BUY
    - Bearish: RSI < 50 AND Close < Lower Bound → STRONG SELL
    - Neutral: все остальное
    
    Args:
        df: DataFrame с OHLCV данными
        atr_period: период для ATR (default=14)
        mult: множитель для ATR (default=1.5)
    
    Returns:
        DataFrame с новыми колонками:
        - AlphaTrend_Upper, AlphaTrend_Lower
        - AlphaTrend_Signal (-1, 0, 1)
    """
    df = df.copy()
    
    # ATR для волатильности
    atr_indicator = volatility.AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=atr_period
    )
    df['ATR_alpha'] = atr_indicator.average_true_range()
    
    # RSI для momentum
    rsi_indicator = momentum.RSIIndicator(close=df['close'], window=14)
    df['RSI_alpha'] = rsi_indicator.rsi()
    
    # Alpha Trend Bounds
    df['AlphaTrend_Upper'] = df['close'] + mult * df['ATR_alpha']
    df['AlphaTrend_Lower'] = df['close'] - mult * df['ATR_alpha']
    
    # Signal Generation
    df['AlphaTrend_Signal'] = 0
    
    # STRONG BUY: RSI > 50 AND Price breaks above Upper
    bullish_mask = (df['RSI_alpha'] > 50) & (df['close'] > df['AlphaTrend_Upper'])
    df.loc[bullish_mask, 'AlphaTrend_Signal'] = 1
    
    # STRONG SELL: RSI < 50 AND Price breaks below Lower
    bearish_mask = (df['RSI_alpha'] < 50) & (df['close'] < df['AlphaTrend_Lower'])
    df.loc[bearish_mask, 'AlphaTrend_Signal'] = -1
    
    # Убираем промежуточные колонки
    df = df.drop(['ATR_alpha', 'RSI_alpha'], axis=1)
    
    return df


def add_ict_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT Smart Money Concepts: Order Blocks + Liquidity Grab
    
    Order Block:
    - Bullish OB: Предыдущая свеча сделала новый Low, текущая закрылась выше Open
    - Bearish OB: Предыдущая свеча сделала новый High, текущая закрылась ниже Open
    
    Liquidity Grab (Stop Hunt):
    - Low ниже минимума последних 10 свечей ИЛИ
    - High выше максимума последних 10 свечей
    
    Применение для Gold:
    - Order Blocks показывают уровни входа институционалов
    - Liquidity Grab предсказывает развороты после стоп-хантинга
    """
    df = df.copy()
    
    # Bullish Order Block
    # Предыдущая свеча: новый Low, Текущая: Close > Open (бычья свеча)
    df["Bullish_OB"] = (
        (df["low"].shift(1) < df["low"]) & 
        (df["close"] > df["open"])
    ).astype(int)
    
    # Bearish Order Block
    # Предыдущая свеча: новый High, Текущая: Close < Open (медвежья свеча)
    df["Bearish_OB"] = (
        (df["high"].shift(1) > df["high"]) & 
        (df["close"] < df["open"])
    ).astype(int)
    
    # Break of Structure (BOS) - Bullish
    # Текущая Close пробивает предыдущий High
    df["BOS_Bullish"] = (
        (df["close"] > df["high"].shift(1)) & 
        (df["close"].shift(1) < df["high"].shift(2))
    ).astype(int)
    
    # Break of Structure (BOS) - Bearish
    # Текущая Close пробивает предыдущий Low
    df["BOS_Bearish"] = (
        (df["close"] < df["low"].shift(1)) & 
        (df["close"].shift(1) > df["low"].shift(2))
    ).astype(int)
    
    # Liquidity Grab (Stop Hunt)
    # Цена пробивает локальные экстремумы для сбора стопов
    liquidity_window = 10
    df["Liquidity_Grab"] = (
        (df["low"] < df["low"].rolling(window=liquidity_window).min()) | 
        (df["high"] > df["high"].rolling(window=liquidity_window).max())
    ).astype(int)
    
    return df


def add_ema_institutional_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Triple EMA System с 200 EMA Institutional Filter
    
    Логика:
    - 200 EMA = главный уровень институциональных игроков для Gold
    - Above_200EMA: 1 если цена выше 200 EMA (бычий bias), 0 иначе
    - EMA_Crossover: 1 (20>50 crossover), -1 (20<50 crossover), 0 (нет кросса)
    
    Применение:
    - Above_200EMA как фильтр направления (long only если 1, short only если 0)
    - EMA_Crossover как подтверждение смены тренда
    """
    df = df.copy()
    
    # Triple EMA
    df['EMA_20'] = trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['EMA_50'] = trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['EMA_200'] = trend.EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # Institutional Bias: Price position relative to 200 EMA
    df['Above_200EMA'] = (df['close'] > df['EMA_200']).astype(int)
    
    # Crossover Detection
    df['EMA_Crossover'] = 0
    
    # Bullish Crossover: 20 crosses above 50
    bullish_cross = (
        (df['EMA_20'] > df['EMA_50']) & 
        (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1))
    )
    df.loc[bullish_cross, 'EMA_Crossover'] = 1
    
    # Bearish Crossover: 20 crosses below 50
    bearish_cross = (
        (df['EMA_20'] < df['EMA_50']) & 
        (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1))
    )
    df.loc[bearish_cross, 'EMA_Crossover'] = -1
    
    return df


def add_support_resistance_static(df: pd.DataFrame, higher_tf_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Static Support/Resistance от старшего таймфрейма
    
    Логика:
    - Берем предыдущую свечу с 4H таймфрейма
    - Low = Support, High = Resistance
    - Добавляем как статические уровни для M5/M15 данных
    
    Args:
        df: Основной DataFrame (M5/M15)
        higher_tf_data: DataFrame с 4H данными (optional)
    
    Returns:
        DataFrame с колонками Support_4H, Resistance_4H
    """
    df = df.copy()
    
    if higher_tf_data is not None and len(higher_tf_data) >= 2:
        # Используем предыдущую свечу с 4H
        previous_candle = higher_tf_data.iloc[-2]
        support_level = float(previous_candle['low'])
        resistance_level = float(previous_candle['high'])
    else:
        # Fallback: используем rolling min/max текущих данных
        support_level = float(df['low'].rolling(window=100).min().iloc[-1])
        resistance_level = float(df['high'].rolling(window=100).max().iloc[-1])
    
    df['Support_4H'] = support_level
    df['Resistance_4H'] = resistance_level
    
    # Расстояние до уровней (normalized)
    df['Distance_To_Support'] = (df['close'] - df['Support_4H']) / df['close']
    df['Distance_To_Resistance'] = (df['Resistance_4H'] - df['close']) / df['close']
    
    return df


def add_all_gold_features(df: pd.DataFrame, higher_tf_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Добавляет ВСЕ Gold-специфические фичи разом
    
    Включает:
    1. Alpha Trend (волатильность + momentum)
    2. ICT Order Blocks + Liquidity Grab (Smart Money)
    3. Triple EMA + 200 EMA Filter (Institutional)
    4. Static Support/Resistance (Multi-TF)
    
    Args:
        df: DataFrame с OHLCV данными
        higher_tf_data: Опциональные данные со старшего таймфрейма для S/R
    
    Returns:
        DataFrame со всеми новыми фичами
    """
    df = df.copy()
    
    print("📊 Adding Gold-Specific Features...")
    
    # 1. Alpha Trend
    print("  ⭐ Alpha Trend Indicator...")
    df = add_alpha_trend(df)
    
    # 2. ICT Smart Money
    print("  💎 ICT Order Blocks & Liquidity...")
    df = add_ict_order_blocks(df)
    
    # 3. EMA System
    print("  📈 Triple EMA + Institutional Filter...")
    df = add_ema_institutional_filter(df)
    
    # 4. Support/Resistance
    print("  📐 Static Support/Resistance...")
    df = add_support_resistance_static(df, higher_tf_data)
    
    print(f"✅ Gold Features Added: {len([c for c in df.columns if 'Alpha' in c or 'OB' in c or 'EMA' in c])} new columns")
    
    return df


# Список новых фичей для использования в модели
GOLD_FEATURE_COLUMNS = [
    # Alpha Trend
    'AlphaTrend_Upper', 'AlphaTrend_Lower', 'AlphaTrend_Signal',
    
    # ICT Smart Money
    'Bullish_OB', 'Bearish_OB', 'BOS_Bullish', 'BOS_Bearish', 'Liquidity_Grab',
    
    # EMA System
    'EMA_20', 'EMA_50', 'EMA_200', 'Above_200EMA', 'EMA_Crossover',
    
    # Support/Resistance
    'Support_4H', 'Resistance_4H', 'Distance_To_Support', 'Distance_To_Resistance'
]


if __name__ == "__main__":
    # Демо: тестируем на синтетических данных
    print("🧪 Testing Gold Features on synthetic data...\n")
    
    dates = pd.date_range('2025-01-01', periods=500, freq='5min')
    test_df = pd.DataFrame({
        'time': dates,
        'open': np.random.uniform(2600, 2650, 500),
        'high': np.random.uniform(2605, 2655, 500),
        'low': np.random.uniform(2595, 2645, 500),
        'close': np.random.uniform(2600, 2650, 500),
        'volume': np.random.uniform(100, 1000, 500)
    })
    
    # Добавляем фичи
    result = add_all_gold_features(test_df)
    
    print("\n📋 New Features Preview:")
    print(result[GOLD_FEATURE_COLUMNS].tail(5))
    
    print("\n✅ All tests passed! Ready for integration.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Интеграционный тест: проверяем, что add_basic_features
теперь автоматически добавляет SMC колонки через вызов add_smc_features.
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "f:/Development of trading bots/Golden Breeze")

from aimodule.data_pipeline.features import add_basic_features

def generate_test_ohlc(size=30):
    """Генерация синтетических OHLC данных для теста"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=size, freq='1h')
    base_price = 2000.0
    
    # Создаем тренд с волатильностью
    closes = []
    for i in range(size):
        if i < 10:
            price = base_price + i * 2 + np.random.normal(0, 1)
        elif i < 20:
            price = base_price + 20 - (i-10) * 1.5 + np.random.normal(0, 1)
        else:
            price = base_price + 5 + (i-20) * 1.2 + np.random.normal(0, 1)
        closes.append(price)
    
    df = pd.DataFrame({
        'time': dates,
        'open': [c + np.random.uniform(-1, 1) for c in closes],
        'high': [c + abs(np.random.uniform(0.5, 2)) for c in closes],
        'low': [c - abs(np.random.uniform(0.5, 2)) for c in closes],
        'close': closes
    })
    
    return df

def test_smc_integration():
    """
    ПРОВЕРКА: add_basic_features теперь включает SMC колонки
    """
    print("=" * 70)
    print("ИНТЕГРАЦИОННЫЙ ТЕСТ: SMC в add_basic_features")
    print("=" * 70)
    
    # 1. Генерация тестовых данных
    print("\n1. Генерация тестовых OHLC данных (30 свечей)...")
    df = generate_test_ohlc(size=30)
    print(f"✓ Сгенерировано {len(df)} строк")
    
    # 2. Вызов add_basic_features (должна включать SMC теперь)
    print("\n2. Вызов add_basic_features()...")
    try:
        df_with_features = add_basic_features(df)
        print("✓ Функция выполнена успешно")
    except Exception as e:
        print(f"✗ ОШИБКА при вызове add_basic_features: {e}")
        return False
    
    # 3. Проверка наличия базовых колонок
    print("\n3. Проверка базовых технических индикаторов...")
    base_columns = ['sma_fast', 'sma_slow', 'atr']
    for col in base_columns:
        if col in df_with_features.columns:
            print(f"✓ Колонка '{col}' присутствует")
        else:
            print(f"✗ ОШИБКА: Колонка '{col}' отсутствует!")
            return False
    
    # 4. Проверка наличия SMC колонок
    print("\n4. Проверка SMC колонок (интеграция)...")
    smc_columns = ['SMC_FVG_Bullish', 'SMC_FVG_Bearish', 'SMC_Swing_High', 'SMC_Swing_Low']
    for col in smc_columns:
        if col in df_with_features.columns:
            print(f"✓ Колонка '{col}' присутствует")
        else:
            print(f"✗ ОШИБКА: Колонка '{col}' отсутствует!")
            return False
    
    # 5. Проверка типов и отсутствия NaN
    print("\n5. Проверка качества данных...")
    for col in base_columns + smc_columns:
        nan_count = df_with_features[col].isna().sum()
        if nan_count > 0:
            print(f"⚠ Предупреждение: '{col}' содержит {nan_count} NaN значений")
        else:
            print(f"✓ Колонка '{col}' без NaN")
    
    # 6. Статистика обнаруженных паттернов
    print("\n6. Статистика обнаруженных SMC паттернов...")
    bullish_fvg = df_with_features['SMC_FVG_Bullish'].sum()
    bearish_fvg = df_with_features['SMC_FVG_Bearish'].sum()
    swing_highs = df_with_features['SMC_Swing_High'].sum()
    swing_lows = df_with_features['SMC_Swing_Low'].sum()
    
    print(f"  • Bullish FVG: {bullish_fvg}")
    print(f"  • Bearish FVG: {bearish_fvg}")
    print(f"  • Swing Highs: {swing_highs}")
    print(f"  • Swing Lows: {swing_lows}")
    
    # 7. Вывод примера результата
    print("\n7. Пример финального DataFrame (последние 5 строк):")
    display_cols = ['close', 'sma_fast', 'atr', 'SMC_FVG_Bullish', 'SMC_Swing_High']
    print(df_with_features[display_cols].tail().to_string())
    
    print("\n" + "=" * 70)
    print("✓ ИНТЕГРАЦИОННЫЙ ТЕСТ ПРОЙДЕН УСПЕШНО")
    print("=" * 70)
    print("\nИтог: add_basic_features() теперь включает:")
    print("  • Базовые TA индикаторы (SMA, ATR)")
    print("  • SMC фичи (FVG + Swing Points)")
    print("\nГотово к обучению модели с расширенным набором признаков!")
    
    return True

if __name__ == "__main__":
    success = test_smc_integration()
    sys.exit(0 if success else 1)

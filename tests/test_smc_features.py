"""
Тестовый скрипт для проверки SMC функций (features_smc.py).

Создает синтетические OHLC данные и проверяет:
- Корректность работы функций
- Наличие ожидаемых колонок
- Отсутствие NaN значений
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корневую директорию проекта в путь
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aimodule.data_pipeline.features_smc import add_smc_features


def generate_synthetic_ohlc(n_rows=20):
    """
    Генерирует синтетические OHLC данные с паттернами для тестирования SMC.
    
    Включает:
    - Сильное движение вверх (для Bullish FVG)
    - Сильное движение вниз (для Bearish FVG)
    - Локальные максимумы и минимумы (для Swing Points)
    """
    np.random.seed(42)
    
    # Базовая цена и волатильность
    base_price = 2000.0
    
    # Создаем паттерн: флэт → резкий рост → флэт → резкое падение → флэт
    prices = []
    
    # Начальный флэт (5 свечей)
    for i in range(5):
        prices.append(base_price + np.random.uniform(-2, 2))
    
    # Резкий рост (3 свечи) - должен создать Bullish FVG
    prices.extend([2005, 2015, 2025])
    
    # Флэт на вершине (3 свечи)
    for i in range(3):
        prices.append(2025 + np.random.uniform(-2, 2))
    
    # Резкое падение (3 свечи) - должен создать Bearish FVG
    prices.extend([2020, 2010, 1995])
    
    # Финальный флэт (6 свечей)
    for i in range(6):
        prices.append(1995 + np.random.uniform(-2, 2))
    
    # Создаем OHLC из цен закрытия
    data = []
    for i, close in enumerate(prices):
        # Генерируем open, high, low на основе close
        open_price = close + np.random.uniform(-1, 1)
        high = max(open_price, close) + abs(np.random.uniform(0, 2))
        low = min(open_price, close) - abs(np.random.uniform(0, 2))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': np.random.randint(100, 1000)
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='5min')
    
    return df


def test_smc_features():
    """
    Основная функция тестирования SMC признаков.
    """
    print("=" * 70)
    print("SMC FEATURES TEST")
    print("=" * 70)
    
    # 1. Генерация тестовых данных
    print("\n1. Генерация синтетических OHLC данных...")
    df = generate_synthetic_ohlc(n_rows=20)
    print(f"   ✓ Создано {len(df)} свечей")
    
    # 2. Применение SMC функций
    print("\n2. Применение SMC функций...")
    df_with_smc = add_smc_features(df.copy())
    print("   ✓ Функция add_smc_features() выполнена успешно")
    
    # 3. Проверка наличия колонок
    print("\n3. Проверка наличия новых колонок...")
    expected_columns = [
        'SMC_FVG_Bullish',
        'SMC_FVG_Bearish',
        'SMC_Swing_High',
        'SMC_Swing_Low'
    ]
    
    missing_columns = []
    for col in expected_columns:
        if col in df_with_smc.columns:
            print(f"   ✓ Колонка '{col}' присутствует")
        else:
            print(f"   ✗ Колонка '{col}' отсутствует!")
            missing_columns.append(col)
    
    if missing_columns:
        print(f"\n   ОШИБКА: Отсутствуют колонки: {missing_columns}")
        return False
    
    # 4. Проверка на NaN значения
    print("\n4. Проверка на NaN значения...")
    nan_counts = {}
    for col in expected_columns:
        nan_count = df_with_smc[col].isna().sum()
        nan_counts[col] = nan_count
        if nan_count > 0:
            print(f"   ✗ Колонка '{col}' содержит {nan_count} NaN значений!")
        else:
            print(f"   ✓ Колонка '{col}' не содержит NaN")
    
    if any(nan_counts.values()):
        print(f"\n   ПРЕДУПРЕЖДЕНИЕ: Найдены NaN значения")
    
    # 5. Статистика по обнаруженным паттернам
    print("\n5. Статистика обнаруженных SMC паттернов...")
    for col in expected_columns:
        count = df_with_smc[col].sum()
        print(f"   • {col}: {count} обнаружено")
    
    # 6. Визуальный вывод последних 10 строк
    print("\n6. Последние 10 строк с SMC признаками:")
    print("-" * 70)
    
    # Выбираем только нужные колонки для отображения
    display_cols = ['close'] + expected_columns
    display_df = df_with_smc[display_cols].tail(10)
    
    # Форматируем вывод
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    pd.set_option('display.width', 100)
    print(display_df.to_string())
    
    # 7. Детальный анализ FVG
    print("\n7. Детальный анализ Fair Value Gaps:")
    print("-" * 70)
    
    bullish_fvg_indices = df_with_smc[df_with_smc['SMC_FVG_Bullish'] == 1].index
    bearish_fvg_indices = df_with_smc[df_with_smc['SMC_FVG_Bearish'] == 1].index
    
    if len(bullish_fvg_indices) > 0:
        print(f"\n   Bullish FVG обнаружены на позициях:")
        for idx in bullish_fvg_indices:
            row_num = df_with_smc.index.get_loc(idx)
            print(f"   - Строка {row_num}: {idx}")
    else:
        print("\n   Bullish FVG не обнаружены")
    
    if len(bearish_fvg_indices) > 0:
        print(f"\n   Bearish FVG обнаружены на позициях:")
        for idx in bearish_fvg_indices:
            row_num = df_with_smc.index.get_loc(idx)
            print(f"   - Строка {row_num}: {idx}")
    else:
        print("\n   Bearish FVG не обнаружены")
    
    # 8. Детальный анализ Swing Points
    print("\n8. Детальный анализ Swing Points:")
    print("-" * 70)
    
    swing_high_indices = df_with_smc[df_with_smc['SMC_Swing_High'] == 1].index
    swing_low_indices = df_with_smc[df_with_smc['SMC_Swing_Low'] == 1].index
    
    if len(swing_high_indices) > 0:
        print(f"\n   Swing High обнаружены на позициях:")
        for idx in swing_high_indices:
            row_num = df_with_smc.index.get_loc(idx)
            high_price = df_with_smc.loc[idx, 'high']
            print(f"   - Строка {row_num}: {idx}, High={high_price:.2f}")
    else:
        print("\n   Swing High не обнаружены")
    
    if len(swing_low_indices) > 0:
        print(f"\n   Swing Low обнаружены на позициях:")
        for idx in swing_low_indices:
            row_num = df_with_smc.index.get_loc(idx)
            low_price = df_with_smc.loc[idx, 'low']
            print(f"   - Строка {row_num}: {idx}, Low={low_price:.2f}")
    else:
        print("\n   Swing Low не обнаружены")
    
    # Финальный результат
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    if not missing_columns and all(v == 0 for v in nan_counts.values()):
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
        print("\nSMC функции работают корректно:")
        print("  • Все ожидаемые колонки созданы")
        print("  • NaN значения отсутствуют")
        print("  • Паттерны корректно обнаруживаются")
        return True
    else:
        print("✗ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        if missing_columns:
            print(f"  • Отсутствующие колонки: {missing_columns}")
        if any(nan_counts.values()):
            print(f"  • NaN значения в колонках: {[k for k, v in nan_counts.items() if v > 0]}")
        return False


if __name__ == "__main__":
    try:
        success = test_smc_features()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

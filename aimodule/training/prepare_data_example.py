# aimodule/training/prepare_data_example.py

"""
Пример подготовки данных XAUUSD в CSV формат для обучения.
Ожидаемый формат CSV: timestamp,open,high,low,close,volume
"""

import pandas as pd


def to_csv(candles, out_path: str):
    df = pd.DataFrame(candles)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # Пример: сохраняем тестовые свечи в data/xauusd_m5.csv
    candles = [
        {"timestamp": "2025-11-30T09:00:00", "open": 2640.0, "high": 2642.0, "low": 2639.0, "close": 2641.5, "volume": 1000.0},
        {"timestamp": "2025-11-30T09:05:00", "open": 2641.5, "high": 2643.0, "low": 2640.5, "close": 2642.0, "volume": 1100.0},
        {"timestamp": "2025-11-30T09:10:00", "open": 2642.0, "high": 2644.5, "low": 2641.5, "close": 2644.0, "volume": 1200.0},
        # ... добавьте реальные исторические данные
    ]
    to_csv(candles, "data/xauusd_m5.csv")
    print("Saved example to data/xauusd_m5.csv")

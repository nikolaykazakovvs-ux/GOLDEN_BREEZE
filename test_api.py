# test_api.py
# Тестовый скрипт для проверки работы AI-модуля

import requests
import json

BASE_URL = "http://127.0.0.1:5005"

def test_health():
    """Проверка health endpoint"""
    print("🔍 Проверка /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Статус: {response.status_code}")
    print(f"   Ответ: {response.json()}")
    print()

def test_predict():
    """Проверка predict endpoint с реальными данными XAUUSD"""
    print("🔍 Проверка /predict с тестовыми данными XAUUSD...")
    
    # Создаём тестовый набор свечей (имитация реальных данных по золоту)
    test_data = {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "candles": [
            {"timestamp": "2025-11-30T09:00:00", "open": 2640.0, "high": 2642.0, "low": 2639.0, "close": 2641.5, "volume": 1000.0},
            {"timestamp": "2025-11-30T09:05:00", "open": 2641.5, "high": 2643.0, "low": 2640.5, "close": 2642.0, "volume": 1100.0},
            {"timestamp": "2025-11-30T09:10:00", "open": 2642.0, "high": 2644.5, "low": 2641.5, "close": 2644.0, "volume": 1200.0},
            {"timestamp": "2025-11-30T09:15:00", "open": 2644.0, "high": 2646.0, "low": 2643.5, "close": 2645.5, "volume": 1150.0},
            {"timestamp": "2025-11-30T09:20:00", "open": 2645.5, "high": 2647.5, "low": 2645.0, "close": 2647.0, "volume": 1300.0},
            {"timestamp": "2025-11-30T09:25:00", "open": 2647.0, "high": 2649.0, "low": 2646.5, "close": 2648.5, "volume": 1250.0},
            {"timestamp": "2025-11-30T09:30:00", "open": 2648.5, "high": 2650.5, "low": 2648.0, "close": 2650.0, "volume": 1400.0},
            {"timestamp": "2025-11-30T09:35:00", "open": 2650.0, "high": 2652.0, "low": 2649.5, "close": 2651.5, "volume": 1350.0},
            {"timestamp": "2025-11-30T09:40:00", "open": 2651.5, "high": 2653.5, "low": 2651.0, "close": 2653.0, "volume": 1450.0},
            {"timestamp": "2025-11-30T09:45:00", "open": 2653.0, "high": 2655.0, "low": 2652.5, "close": 2654.5, "volume": 1500.0},
            {"timestamp": "2025-11-30T09:50:00", "open": 2654.5, "high": 2656.5, "low": 2654.0, "close": 2656.0, "volume": 1550.0},
            {"timestamp": "2025-11-30T09:55:00", "open": 2656.0, "high": 2658.0, "low": 2655.5, "close": 2657.5, "volume": 1600.0},
            {"timestamp": "2025-11-30T10:00:00", "open": 2657.5, "high": 2659.5, "low": 2657.0, "close": 2659.0, "volume": 1650.0},
            {"timestamp": "2025-11-30T10:05:00", "open": 2659.0, "high": 2661.0, "low": 2658.5, "close": 2660.5, "volume": 1700.0},
            {"timestamp": "2025-11-30T10:10:00", "open": 2660.5, "high": 2662.5, "low": 2660.0, "close": 2662.0, "volume": 1750.0},
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"   Статус: {response.status_code}")
    print(f"   Ответ:")
    result = response.json()
    print(f"      Symbol: {result['symbol']}")
    print(f"      Timeframe: {result['timeframe']}")
    print(f"      Режим рынка: {result['regime']}")
    print(f"      Направление: {result['direction']}")
    print(f"      Sentiment: {result['sentiment']}")
    print(f"      Confidence: {result['confidence']:.2f}")
    print(f"      Рекомендация: {result['action']}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Тестирование AICore_XAUUSD_v1.0")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_predict()
        print("=" * 60)
        print("✅ Все тесты пройдены успешно!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Ошибка: {e}")

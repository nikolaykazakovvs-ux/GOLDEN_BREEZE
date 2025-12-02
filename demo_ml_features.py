# demo_ml_features.py
"""
Демонстрация возможностей Golden Breeze v2.0 с ML-моделями
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5005"

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def demo_health():
    print_header("🏥 Health Check")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")

def demo_minimal_predict():
    print_header("📊 Test 1: Minimal Data (2 candles)")
    print("Проверка работы с минимальным набором данных...")
    
    body = {
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "candles": [
            {"timestamp": "2025-11-30T10:00:00", "open": 2650.0, "high": 2652.0, "low": 2649.0, "close": 2651.0, "volume": 1000.0},
            {"timestamp": "2025-11-30T10:05:00", "open": 2651.0, "high": 2653.0, "low": 2650.5, "close": 2652.5, "volume": 1100.0},
        ]
    }
    
    r = requests.post(f"{BASE_URL}/predict", json=body)
    result = r.json()
    
    print(f"📈 Symbol: {result['symbol']}")
    print(f"⏰ Timeframe: {result['timeframe']}")
    print(f"🎯 Market Regime: {result['regime']}")
    print(f"➡️  Direction: {result['direction']}")
    print(f"💭 Sentiment: {result['sentiment']:.2f}")
    print(f"🎲 Confidence: {result['confidence']:.2%}")
    print(f"🚦 Action: {result['action'].upper()}")
    
    print("\n💡 Анализ:")
    if result['regime'] == 'unknown':
        print("   - Недостаточно данных для определения режима")
    print(f"   - Модель направления: {'LSTM' if result['confidence'] > 0.5 else 'Fallback (momentum)'}")
    print(f"   - Рекомендация: {_explain_action(result['action'])}")

def demo_trending_market():
    print_header("📈 Test 2: Strong Uptrend (60 candles)")
    print("Имитация восходящего тренда...")
    
    # Генерируем восходящий тренд
    candles = []
    base_price = 2640.0
    for i in range(60):
        price = base_price + i * 0.5  # Рост на 0.5 за свечу
        candles.append({
            "timestamp": f"2025-11-30T{9 + i//12:02d}:{(i%12)*5:02d}:00",
            "open": price - 0.2,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000.0 + i * 10
        })
    
    body = {"symbol": "XAUUSD", "timeframe": "M5", "candles": candles}
    r = requests.post(f"{BASE_URL}/predict", json=body)
    result = r.json()
    
    print(f"📈 Symbol: {result['symbol']}")
    print(f"🎯 Market Regime: {result['regime']}")
    print(f"➡️  Direction: {result['direction']}")
    print(f"💭 Sentiment: {result['sentiment']:.2f}")
    print(f"🎲 Confidence: {result['confidence']:.2%}")
    print(f"🚦 Action: {result['action'].upper()}")
    
    print("\n💡 Анализ:")
    print(f"   - Режим: {_explain_regime(result['regime'])}")
    print(f"   - Направление: {_explain_direction(result['direction'])}")
    print(f"   - Sentiment: {_explain_sentiment(result['sentiment'])}")
    print(f"   - Уверенность: {_explain_confidence(result['confidence'])}")
    print(f"   - Итоговое решение: {_explain_action(result['action'])}")

def demo_ranging_market():
    print_header("〰️  Test 3: Ranging Market (40 candles)")
    print("Имитация бокового движения (флэт)...")
    
    # Генерируем флэт с небольшими колебаниями
    candles = []
    base_price = 2650.0
    for i in range(40):
        import math
        oscillation = math.sin(i * 0.5) * 2.0  # Колебания ±2
        price = base_price + oscillation
        candles.append({
            "timestamp": f"2025-11-30T{10 + i//12:02d}:{(i%12)*5:02d}:00",
            "open": price - 0.3,
            "high": price + 0.8,
            "low": price - 0.8,
            "close": price,
            "volume": 1000.0
        })
    
    body = {"symbol": "XAUUSD", "timeframe": "M5", "candles": candles}
    r = requests.post(f"{BASE_URL}/predict", json=body)
    result = r.json()
    
    print(f"📈 Symbol: {result['symbol']}")
    print(f"🎯 Market Regime: {result['regime']}")
    print(f"➡️  Direction: {result['direction']}")
    print(f"💭 Sentiment: {result['sentiment']:.2f}")
    print(f"🎲 Confidence: {result['confidence']:.2%}")
    print(f"🚦 Action: {result['action'].upper()}")
    
    print("\n💡 Анализ:")
    print(f"   - В RANGE рекомендуется избегать входов")
    print(f"   - Sentiment слабый → дополнительная причина для HOLD")

def demo_volatile_market():
    print_header("⚡ Test 4: Volatile Market (50 candles)")
    print("Имитация высокой волатильности...")
    
    # Генерируем волатильный рынок
    candles = []
    base_price = 2650.0
    for i in range(50):
        import random
        volatility = random.uniform(-5, 5)
        price = base_price + volatility
        candles.append({
            "timestamp": f"2025-11-30T{11 + i//12:02d}:{(i%12)*5:02d}:00",
            "open": price - random.uniform(0, 2),
            "high": price + random.uniform(0, 3),
            "low": price - random.uniform(0, 3),
            "close": price,
            "volume": 1000.0 + random.uniform(-200, 200)
        })
    
    body = {"symbol": "XAUUSD", "timeframe": "M5", "candles": candles}
    r = requests.post(f"{BASE_URL}/predict", json=body)
    result = r.json()
    
    print(f"📈 Symbol: {result['symbol']}")
    print(f"🎯 Market Regime: {result['regime']}")
    print(f"➡️  Direction: {result['direction']}")
    print(f"💭 Sentiment: {result['sentiment']:.2f}")
    print(f"🎲 Confidence: {result['confidence']:.2%}")
    print(f"🚦 Action: {result['action'].upper()}")
    
    print("\n💡 Анализ:")
    print(f"   - В VOLATILE повышается минимальный порог confidence")
    print(f"   - Требуется confidence > 0.35 для входа")

def _explain_regime(regime):
    explanations = {
        "trend_up": "Восходящий тренд (SMA fast > SMA slow)",
        "trend_down": "Нисходящий тренд (SMA fast < SMA slow)",
        "range": "Боковое движение (SMA примерно равны)",
        "volatile": "Высокая волатильность (высокий ATR)",
        "unknown": "Недостаточно данных для определения"
    }
    return explanations.get(regime, regime)

def _explain_direction(direction):
    explanations = {
        "long": "Ожидается движение вверх",
        "short": "Ожидается движение вниз",
        "flat": "Ожидается боковое движение"
    }
    return explanations.get(direction, direction)

def _explain_sentiment(sentiment):
    if sentiment > 0.3:
        return "Очень позитивный"
    elif sentiment > 0.1:
        return "Позитивный"
    elif sentiment > -0.1:
        return "Нейтральный"
    elif sentiment > -0.3:
        return "Негативный"
    else:
        return "Очень негативный"

def _explain_confidence(confidence):
    if confidence > 0.7:
        return "Очень высокая"
    elif confidence > 0.5:
        return "Высокая"
    elif confidence > 0.3:
        return "Средняя"
    elif confidence > 0.2:
        return "Низкая"
    else:
        return "Очень низкая"

def _explain_action(action):
    explanations = {
        "enter_long": "Рекомендуется ПОКУПКА (long позиция)",
        "enter_short": "Рекомендуется ПРОДАЖА (short позиция)",
        "hold": "Рекомендуется УДЕРЖИВАТЬ / не входить (недостаточно сигналов)",
        "skip": "Рекомендуется ПРОПУСТИТЬ (плохие условия)",
        "exit": "Рекомендуется ВЫЙТИ из позиции"
    }
    return explanations.get(action, action)

if __name__ == "__main__":
    print_header("🚀 Golden Breeze v2.0 - ML Features Demo")
    print("Демонстрация работы AI-моделей и комбинированной логики принятия решений")
    
    try:
        demo_health()
        demo_minimal_predict()
        demo_trending_market()
        demo_ranging_market()
        demo_volatile_market()
        
        print_header("✅ Демонстрация завершена!")
        print("\n📚 Дополнительная информация:")
        print("   - TRAINING_GUIDE.md - как обучить модели")
        print("   - ML_INTEGRATION_REPORT.md - отчёт об интеграции")
        print("   - START_HERE.md - быстрый старт")
        print("\n🎯 Следующий шаг: обучите модели на реальных данных!")
        print("   python -m aimodule.training.train_direction_lstm")
        print("   python -m aimodule.training.train_regime_cluster")
        print("   python -m aimodule.training.build_sentiment_lexicon")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Убедитесь, что сервер запущен:")
        print("   .\\run_server.ps1")

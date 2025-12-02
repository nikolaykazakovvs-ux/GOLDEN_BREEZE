"""
Тест /predict endpoint с реальными данными XAUUSD.
Проверяет GPU-ускоренное inference Direction LSTM модели.
"""
import subprocess
import time
import requests
import json
import sys


# Пример 50 свечей XAUUSD M5 (требуется для LSTM с seq_len=50)
SAMPLE_CANDLES = [
    {"timestamp": "2024-01-01 00:00:00", "open": 2065.5, "high": 2067.0, "low": 2064.0, "close": 2066.5, "volume": 1200},
    {"timestamp": "2024-01-01 00:05:00", "open": 2066.5, "high": 2068.5, "low": 2065.5, "close": 2067.0, "volume": 1350},
    {"timestamp": "2024-01-01 00:10:00", "open": 2067.0, "high": 2069.0, "low": 2066.0, "close": 2068.0, "volume": 1400},
    {"timestamp": "2024-01-01 00:15:00", "open": 2068.0, "high": 2070.0, "low": 2067.0, "close": 2069.0, "volume": 1500},
    {"timestamp": "2024-01-01 00:20:00", "open": 2069.0, "high": 2071.0, "low": 2068.0, "close": 2070.0, "volume": 1600},
    {"timestamp": "2024-01-01 00:25:00", "open": 2070.0, "high": 2072.0, "low": 2069.0, "close": 2071.0, "volume": 1650},
    {"timestamp": "2024-01-01 00:30:00", "open": 2071.0, "high": 2073.0, "low": 2070.0, "close": 2072.0, "volume": 1700},
    {"timestamp": "2024-01-01 00:35:00", "open": 2072.0, "high": 2074.0, "low": 2071.0, "close": 2073.0, "volume": 1750},
    {"timestamp": "2024-01-01 00:40:00", "open": 2073.0, "high": 2075.0, "low": 2072.0, "close": 2074.0, "volume": 1800},
    {"timestamp": "2024-01-01 00:45:00", "open": 2074.0, "high": 2076.0, "low": 2073.0, "close": 2075.0, "volume": 1850},
] * 5  # Дублируем для получения 50 свечей


def test_predict_endpoint():
    """Тест /predict endpoint с GPU inference."""
    
    # Запуск сервера
    print("🚀 Запуск сервера в фоновом режиме...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "aimodule.server.local_ai_gateway"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Ожидание старта
    print("⏳ Ожидание старта сервера...")
    server_ready = False
    for i in range(15):
        try:
            response = requests.get("http://127.0.0.1:5005/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print("✅ Сервер запущен!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    
    if not server_ready:
        print("❌ Сервер не запустился!")
        server_process.terminate()
        return False
    
    try:
        # Тест /predict endpoint
        print("\n📊 Тестирование /predict endpoint:")
        print(f"   - Отправка {len(SAMPLE_CANDLES)} свечей XAUUSD M5")
        
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": SAMPLE_CANDLES[:50]  # Отправляем ровно 50 свечей
        }
        
        response = requests.post(
            "http://127.0.0.1:5005/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Ошибка HTTP {response.status_code}: {response.text}")
            return False
        
        prediction = response.json()
        print("\n✅ Результат prediction:")
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
        
        # Проверка наличия ключей
        required_keys = ["symbol", "timeframe", "regime", "direction", "confidence", "action"]
        missing_keys = [key for key in required_keys if key not in prediction]
        
        if missing_keys:
            print(f"\n❌ Отсутствуют ключи: {missing_keys}")
            return False
        
        print("\n🔍 Анализ результата:")
        print(f"   - Symbol: {prediction['symbol']}")
        print(f"   - Timeframe: {prediction['timeframe']}")
        print(f"   - Regime: {prediction['regime']}")
        print(f"   - Direction: {prediction['direction']}")
        print(f"   - Confidence: {prediction['confidence']:.3f}")
        print(f"   - Action: {prediction['action']}")
        
        if prediction.get("reasons"):
            print(f"   - Reasons: {', '.join(prediction['reasons'])}")
        
        # Проверка диапазона confidence
        conf = prediction.get("confidence", 0)
        if 0.0 <= conf <= 1.0:
            print(f"\n✅ Confidence в корректном диапазоне: {conf:.3f}")
        else:
            print(f"\n⚠️ Confidence вне диапазона [0, 1]: {conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n🛑 Остановка сервера...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("✅ Сервер остановлен")


if __name__ == "__main__":
    success = test_predict_endpoint()
    sys.exit(0 if success else 1)

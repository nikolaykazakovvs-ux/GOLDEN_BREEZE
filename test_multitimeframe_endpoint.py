"""
Тест /predict_multitimeframe endpoint для batch prediction.
Отправляет данные для M5, M15, H1, H4 одновременно и получает прогнозы.
"""
import subprocess
import time
import requests
import json
import sys


# Генерация тестовых данных для разных таймфреймов
def generate_candles(count, base_price=2065.0, trend="up"):
    """Генерация синтетических свечей с трендом."""
    candles = []
    price = base_price
    increment = 0.5 if trend == "up" else -0.5
    
    for i in range(count):
        open_price = price
        close_price = price + increment
        high_price = max(open_price, close_price) + abs(increment) * 0.5
        low_price = min(open_price, close_price) - abs(increment) * 0.5
        
        candles.append({
            "timestamp": f"2024-01-01 {i//12:02d}:{(i%12)*5:02d}:00",
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": 1000 + i * 10
        })
        
        price = close_price
    
    return candles


def test_multitimeframe_endpoint():
    """Тест /predict_multitimeframe endpoint."""
    
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
        # Подготовка данных для multitimeframe prediction
        print("\n📊 Подготовка данных для batch prediction:")
        
        timeframes_data = {
            "M5": generate_candles(50, base_price=2065.0, trend="up"),
            "M15": generate_candles(50, base_price=2065.0, trend="up"),
            "H1": generate_candles(50, base_price=2064.0, trend="down"),
            "H4": generate_candles(50, base_price=2063.0, trend="down"),
        }
        
        for tf, candles in timeframes_data.items():
            print(f"   - {tf}: {len(candles)} свечей")
        
        payload = {
            "symbol": "XAUUSD",
            "timeframes_data": timeframes_data
        }
        
        # Отправка запроса
        print("\n🚀 Отправка batch request на /predict_multitimeframe...")
        response = requests.post(
            "http://127.0.0.1:5005/predict_multitimeframe",
            json=payload,
            timeout=30  # Увеличенный таймаут для batch processing
        )
        
        if response.status_code != 200:
            print(f"❌ Ошибка HTTP {response.status_code}: {response.text}")
            return False
        
        results = response.json()
        print("\n✅ Результаты batch prediction:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
        # Анализ результатов для каждого таймфрейма
        print("\n🔍 Детальный анализ:")
        for tf in ["M5", "M15", "H1", "H4"]:
            if tf not in results:
                print(f"\n❌ Отсутствует результат для {tf}")
                continue
            
            pred = results[tf]
            if "error" in pred:
                print(f"\n❌ {tf}: Ошибка - {pred['error']}")
                continue
            
            print(f"\n✅ {tf}:")
            print(f"   - Regime: {pred.get('regime')}")
            print(f"   - Direction: {pred.get('direction')}")
            print(f"   - Confidence: {pred.get('confidence', 0):.3f}")
            print(f"   - Action: {pred.get('action')}")
            if pred.get('reasons'):
                print(f"   - Reasons: {', '.join(pred['reasons'])}")
        
        # Проверка, что все таймфреймы вернули результаты
        missing_tfs = [tf for tf in ["M5", "M15", "H1", "H4"] if tf not in results or "error" in results[tf]]
        if missing_tfs:
            print(f"\n⚠️ Не удалось получить результаты для: {', '.join(missing_tfs)}")
            return False
        
        print("\n✅ Все таймфреймы успешно обработаны!")
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
    success = test_multitimeframe_endpoint()
    sys.exit(0 if success else 1)

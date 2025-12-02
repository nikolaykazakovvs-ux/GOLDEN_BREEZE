"""
Тест GPU для AICore сервера.
Запускает сервер в отдельном процессе и проверяет GPU через /health endpoint.
"""
import subprocess
import time
import requests
import json
import sys


def test_gpu_support():
    """Запуск сервера и проверка GPU support."""
    
    # Запуск сервера в отдельном процессе
    print("🚀 Запуск сервера в фоновом режиме...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "aimodule.server.local_ai_gateway"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Ждём старта сервера (до 15 секунд)
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
        print(f"   {i+1}/15 секунд...")
    
    if not server_ready:
        print("❌ Сервер не запустился за 15 секунд!")
        server_process.terminate()
        return False
    
    try:
        # Проверка /health endpoint
        print("\n📊 Тестирование /health endpoint:")
        response = requests.get("http://127.0.0.1:5005/health", timeout=3)
        health_data = response.json()
        
        print(json.dumps(health_data, indent=2, ensure_ascii=False))
        
        # Проверка GPU
        print("\n🔍 Проверка GPU:")
        if health_data.get("device") == "cuda":
            print("✅ CUDA устройство: cuda")
        else:
            print(f"❌ Устройство: {health_data.get('device')} (ожидалось cuda)")
            
        if health_data.get("use_gpu") is True:
            print("✅ GPU включен: True")
        else:
            print(f"❌ GPU: {health_data.get('use_gpu')} (ожидалось True)")
        
        # Проверка метаданных модели
        if "direction_model" in health_data:
            print("\n📈 Метаданные Direction Model:")
            for key, value in health_data["direction_model"].items():
                print(f"   - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return False
    
    finally:
        print("\n🛑 Остановка сервера...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("✅ Сервер остановлен")


if __name__ == "__main__":
    success = test_gpu_support()
    sys.exit(0 if success else 1)

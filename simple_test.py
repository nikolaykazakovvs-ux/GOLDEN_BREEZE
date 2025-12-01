# simple_test.py
import requests

print("Проверка /health...")
try:
    response = requests.get("http://127.0.0.1:5005/health", timeout=5)
    print(f"Статус: {response.status_code}")
    print(f"Ответ: {response.json()}")
except Exception as e:
    print(f"Ошибка: {e}")

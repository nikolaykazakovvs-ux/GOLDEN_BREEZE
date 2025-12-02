# Golden Breeze v3.0 — AI Trading Core

Production-ready локальное AI-ядро для сигналов рынка (без торговой стратегии).

## Содержание
- [Мультиагентный протокол](#мультиагентный-протокол)
- [Быстрый self-check окружения](#быстрый-self-check-окружения)
- [Запуск локального AI-сервера](#запуск-локального-ai-сервера)
- [Обучение моделей](#обучение-моделей)
- [Интеграция: AI Connector](#интеграция-ai-connector)
- [Feedback и self-learning](#feedback-и-self-learning)

## Мультиагентный протокол
Актуальный протокол и стандарты разработки находятся в:

- `AGENTS_GOLDEN_BREEZE.md`

Обязателен к прочтению для всех агентов/инструментов.

## Быстрый self-check окружения

Ниже — проверка базового окружения в PowerShell (Windows). Выполняйте команды в корне проекта.

```powershell
# 1) Версия Python
python --version

# 2) Виртуальное окружение (опционально)
# Если venv отсутствует — создайте и активируйте
python -m venv venv
./venv/Scripts/Activate.ps1

# 3) Зависимости (установите при необходимости)
pip install -U pip
pip install torch transformers fastapi pydantic numpy pandas scikit-learn uvicorn ta

# 4) Проверка GPU
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
PY
```

## Запуск локального AI-сервера

```powershell
# Запуск FastAPI сервера
uvicorn aimodule.server.local_ai_gateway:app --host 127.0.0.1 --port 5005

# Проверка состояния
Invoke-WebRequest -Uri "http://127.0.0.1:5005/health" -UseBasicParsing
```

## Обучение моделей

Regime Model v3:
```powershell
python -m aimodule.training.train_regime_cluster --data "data/xauusd_m5.csv" --model kmeans --clusters 4
```

Direction Model v3:
```powershell
python -m aimodule.training.train_direction_model --data "data/xauusd_history.csv" --seq-len 50 --epochs 20 --epsilon 0.0005 --patience 5
```

## Интеграция: AI Connector

Пример использования Python-клиента:
```python
from aimodule.connector import GoldenBreezeClient
client = GoldenBreezeClient()

# Предсказание
resp = client.predict(symbol="XAUUSD", timeframe="M5", candles=[...])

# Feedback (после закрытия сделки)
client.send_feedback({
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "direction": resp["direction"],
  "action": resp["action"],
  "pnl": 15.0,
  "regime": resp["regime"],
  "sentiment": resp["sentiment"],
  "confidence": resp["confidence"],
  "timestamp": "2025-12-01T12:00:00Z"
})
```

## Feedback и self-learning

- Endpoint: `POST /feedback` — принимает результат сделки и запускает online-адаптацию порогов.
- Данные сохраняются в `data/trade_feedback.csv`.
- Динамические пороги — `data/config_dynamic.json`.

Примечание: Golden Breeze не содержит торговой стратегии. Он выдаёт сигналы (regime, direction, sentiment, confidence, action, reasons), а решение об исполнении принимает внешняя система.

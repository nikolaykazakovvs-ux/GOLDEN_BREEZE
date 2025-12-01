# Golden Breeze - AICore_XAUUSD_v2.0

Локальный AI-модуль для генерации торговых сигналов по золоту (XAUUSD).

**🎉 Версия 2.0 - Full AI Suite:**
- ✅ **RegimeMLModel**: ML-кластеризация режимов (KMeans/GMM)
- ✅ **DirectionLSTM**: Улучшенная LSTM для прогноза направления
- ✅ **HF Sentiment**: Локальная HuggingFace модель для анализа настроений
- ✅ **Enhanced Ensemble**: Продвинутая логика принятия решений с объяснениями
- ✅ **Training Pipeline**: Полный набор скриптов для обучения моделей

## 🚀 Три уровня AI

### 1. Regime ML Model (Market Regime Detector)
- **Технология**: KMeans или GaussianMixture (scikit-learn)
- **Признаки**: returns, ATR, SMA slope, volatility
- **Обучение**: `python -m aimodule.training.train_regime_model`
- **Модель**: `models/regime_ml.pkl`
- **Fallback**: RegimeClusterModel (простая кластеризация)

### 2. Direction LSTM Model
- **Технология**: PyTorch LSTM (2 слоя, 64 hidden units)
- **Признаки**: close, returns, sma_fast, sma_slow, atr, volume
- **Обучение**: `python -m aimodule.training.train_direction_model`
- **Модель**: `models/direction_lstm.pt`
- **Fallback**: Базовая LSTM или momentum

### 3. Sentiment Engine
- **Уровень 1**: HuggingFace модель (twitter-roberta-base-sentiment)
- **Уровень 2**: Lexicon модель (word-weight dictionary)
- **Уровень 3**: Regime-based baseline
- **Источники**: Mock news (расширяемо до NewsAPI, RSS, Twitter)

## Структура проекта

```
Golden Breeze/
├── requirements.txt
├── run_install_ai.ps1          # Установка зависимостей
├── run_tests.ps1               # Запуск тестов
├── test_ai_core.py             # Тесты AI-ядра
├── aimodule/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regime_model.py          # Базовая кластеризация
│   │   ├── regime_ml_model.py       # 🆕 ML кластеризация (KMeans/GMM)
│   │   ├── direction_model.py       # Базовая LSTM
│   │   ├── direction_lstm_model.py  # 🆕 Улучшенная LSTM
│   │   ├── sentiment_model.py       # Lexicon модель
│   │   ├── sentiment_hf_model.py    # 🆕 HuggingFace модель
│   │   └── sentiment_engine.py      # 🆕 Unified sentiment
│   ├── sentiment_source/            # 🆕 Источники новостей
│   │   ├── __init__.py
│   │   └── news_source.py
│   ├── training/                    # 🆕 Скрипты обучения
│   │   ├── __init__.py
│   │   ├── train_regime_model.py
│   │   ├── train_direction_model.py
│   │   ├── train_regime_cluster.py  # Legacy
│   │   └── train_direction_lstm.py  # Legacy
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict_regime.py
│   │   ├── predict_direction.py
│   │   └── combine_signals.py       # 🆕 Enhanced с reasons
│   └── server/
│       ├── __init__.py
│       └── local_ai_gateway.py      # v2.0 API
└── README.md
```

## Установка

### Быстрая установка (Windows)
```powershell
.\run_install_ai.ps1
```

### Ручная установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Зависимости включают:**
- FastAPI, uvicorn - веб-сервер
- PyTorch - LSTM модели
- transformers - HuggingFace sentiment
- scikit-learn - ML кластеризация
- pandas, numpy - обработка данных
- ta - технические индикаторы
- pytest - тестирование

## Запуск сервера

```bash
uvicorn aimodule.server.local_ai_gateway:app --host 127.0.0.1 --port 5005 --reload
```

## Проверка работы

Проверьте здоровье сервера:
```bash
curl http://127.0.0.1:5005/health
```

## API Endpoints

### GET /health
Проверка состояния сервера

### POST /predict
Получение торгового сигнала

**Request:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "candles": [
    {
      "timestamp": "2025-11-30T10:00:00",
      "open": 2650.5,
      "high": 2652.0,
      "low": 2649.0,
      "close": 2651.5,
      "volume": 1000.0
    }
  ]
}
```

**Response:**
```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "regime": "trend_up",
  "direction": "long",
  "sentiment": 0.3,
  "confidence": 0.75,
  "action": "enter_long",
  "reasons": [
    "Strong uptrend detected (regime: trend_up)",
    "Direction model predicts LONG (confidence: 75%)",
    "Positive sentiment (0.30) supports entry"
  ]
}
```

## 🎓 How to Retrain Models

### Подготовка данных
1. Создайте папку `data/` в корне проекта
2. Поместите файл `xauusd_history.csv` с форматом:
```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00,2000.0,2010.0,1995.0,2005.0,1000.0
2024-01-01T00:05:00,2005.0,2015.0,2000.0,2010.0,1200.0
...
```
**Рекомендуется**: минимум 10,000 свечей для качественного обучения

### Обучение Regime Model
```bash
python -m aimodule.training.train_regime_model
```
- **Метод**: KMeans (по умолчанию) или GMM
- **Признаки**: returns, ATR, SMA slope, volatility
- **Выход**: `models/regime_ml.pkl`

### Обучение Direction Model
```bash
python -m aimodule.training.train_direction_model
```
- **Архитектура**: LSTM (2 слоя, 64 hidden units)
- **Окно**: 100 свечей
- **Выход**: `models/direction_lstm.pt`
- **Эпохи**: 10 (можно увеличить до 20-30)

### После обучения
```bash
# Перезапустите сервер для загрузки новых моделей
python -m aimodule.server.local_ai_gateway
```

## 🧪 Тестирование

```powershell
.\run_tests.ps1
```

Или вручную:
```bash
pytest test_ai_core.py -v
```

**Тесты проверяют:**
- Health endpoint
- Predict endpoint с разными сценариями
- Валидацию всех полей ответа
- Диапазоны значений (sentiment, confidence)
- Обработку ошибок

## Возможности

### 🎯 Текущие
- **Определение режима рынка**: ML кластеризация (KMeans/GMM) → trend_up/trend_down/range/volatile
- **Прогноз направления**: Улучшенная LSTM с 6 признаками → long/short/flat + confidence
- **Анализ настроений**: HuggingFace transformer + lexicon + regime fallback → [-1, 1]
- **Умные решения**: Ensemble логика с 8 правилами и объяснениями → enter_long/enter_short/hold/skip
- **Graceful degradation**: Все модели имеют fallback при отсутствии обученных весов

### 🔮 Дальнейшее развитие
- Интеграция с real-time news API (NewsAPI, Reuters RSS)
- FinRL для reinforcement learning
- Attention mechanisms для LSTM
- Multi-timeframe analysis
- Risk management модуль
- Связь с PropFirmHybridEngine / Bot Studio

## 📚 Документация

- **TRAINING_GUIDE.md** - детальное руководство по обучению моделей
- **ML_INTEGRATION_REPORT.md** - технический отчёт о ML компонентах
- **DEPLOYMENT_REPORT.md** - информация о деплойменте
- **QUICK_START.md** - быстрый старт

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────┐
│                 FastAPI Gateway                     │
│              (local_ai_gateway.py)                  │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │ Regime │  │Direction│  │Sentiment │
    │   ML   │  │  LSTM  │  │  Engine  │
    └────┬───┘  └───┬────┘  └────┬─────┘
         │          │            │
         └──────────┼────────────┘
                    ▼
            ┌──────────────┐
            │   Ensemble   │
            │decide_action │
            └──────────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Action +   │
            │   Reasons    │
            └──────────────┘
```

## 🔧 Конфигурация

Настройки в `aimodule/config.py`:
- `MODELS_DIR` - директория моделей
- `REGIME_MODEL_PATH` - путь к regime модели
- `DIRECTION_MODEL_PATH` - путь к direction модели
- `SENTIMENT_MODEL_PATH` - путь к sentiment лексикону

## 📞 Support

- Issues: GitHub Issues
- Email: support@goldenbreeze.ai (placeholder)
- Docs: См. TRAINING_GUIDE.md, ML_INTEGRATION_REPORT.md

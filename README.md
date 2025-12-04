# 🥇 Golden Breeze - AI Trading System for XAUUSD

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Локальный AI-модуль для генерации торговых сигналов по золоту (XAUUSD) с мультитаймфреймовой логикой и полным конвейером обучения.

## 🚀 Model v3 Performance (Production-Ready)

| Метрика | Значение |
|---------|----------|
| **Test MCC** | 0.7513 |
| **Accuracy** | 87.59% |
| **Backtest ROI** | +65.76% |
| **Profit Factor** | 1.12 |

**Период обучения**: 6 месяцев (Jun-Dec 2025) | **27,521 последовательностей**

---

## ✨ Ключевые возможности

### 🧠 AI Models
| Модель | Архитектура | Статус |
|--------|-------------|--------|
| **Direction LSTM v3** | 2-Layer LSTM, 32 Gold features | ✅ Production |
| **Fusion Transformer v4** | Dual-Stream Sliding-Patch | 🔧 Development |
| **Regime ML** | KMeans/GMM clustering | ✅ Ready |
| **Sentiment Engine** | HuggingFace + Lexicon | ✅ Ready |

### 📊 Features
- ✅ **Gold-Optimized Features**: 32 специализированных признаков для XAUUSD
- ✅ **Smart Money Concepts (SMC)**: Order Blocks, FVG, Market Structure
- ✅ **GPU Acceleration**: CUDA 12.4, RTX 3070 (2.7-45x speedup)
- ✅ **Multitimeframe Logic**: M5/M15/H1/H4 dynamic selection
- ✅ **MT5 Integration**: Real-time data from MetaTrader 5
- ✅ **Risk Management**: ATR-based SL/TP, drawdown limits

### 🆕 v4 Fusion Transformer (In Development)
- 🔹 Sliding Patch Encoding (overlapping patches)
- 🔹 Dual-Stream Processing (M5 + H1)
- 🔹 SMC Embedding Injection
- 🔹 Gated Cross-Attention Fusion
- 🔹 ~1M parameters, SOTA architecture

## 🚀 AI Models Overview

### 1. Direction LSTM v3 (Production) ⭐
- **Архитектура**: 2-Layer LSTM, 64 hidden units, dropout 0.3
- **Признаки**: 32 (15 базовых + 17 Gold-специфичных)
- **Обучение**: 6 месяцев данных, 27K последовательностей
- **Модель**: `models/direction_lstm_gold_v3.pt` (58K параметров)
- **Результат**: MCC 0.7513, Accuracy 87.59%

### 2. Fusion Transformer v4 (Development) 🆕
- **Архитектура**: Dual-Stream Sliding-Patch Transformer
- **Fast Stream**: M5 (200 bars → 24 patches)
- **Slow Stream**: H1 (50 bars) + SMC tokens
- **Fusion**: Gated Cross-Attention (learnable α)
- **Модель**: `aimodule/models/v4_transformer/` (~1M параметров)

### 3. Regime ML Model
- **Технология**: KMeans/GaussianMixture (scikit-learn)
- **Признаки**: returns, ATR, SMA slope, volatility
- **Модель**: `models/regime_ml.pkl`

### 4. Sentiment Engine
- **Уровень 1**: HuggingFace (twitter-roberta-base-sentiment)
- **Уровень 2**: Lexicon model
- **Уровень 3**: Regime-based fallback

## 📁 Project Structure

```
Golden Breeze/
├── aimodule/                    # AI Core
│   ├── models/
│   │   ├── direction_lstm_model.py    # LSTM v3
│   │   └── v4_transformer/            # 🆕 Fusion Transformer v4
│   │       ├── config.py              # V4Config
│   │       ├── embeddings.py          # SlidingPatchEmbed, SMCEmbed
│   │       ├── fusion.py              # GatedCrossAttention
│   │       └── model.py               # GoldenBreezeFusionV4
│   ├── data_pipeline/
│   │   ├── features.py                # Base features
│   │   ├── features_gold.py           # Gold-specific features
│   │   └── features_smc.py            # SMC features
│   ├── training/                      # Training scripts
│   ├── inference/                     # Prediction modules
│   └── server/                        # FastAPI gateway
├── strategy/                          # Hybrid Strategy v1.1
│   ├── hybrid_strategy.py             # Main strategy class
│   ├── timeframe_selector.py          # Dynamic TF selection
│   ├── risk_manager.py                # Risk management
│   └── backtest_engine.py             # Backtesting
├── models/                            # Trained models
│   ├── direction_lstm_gold_v3.pt      # ⭐ Production model
│   └── direction_lstm_gold_v3.json    # Metadata
├── data/
│   ├── raw/XAUUSD/                    # MT5 exported data
│   ├── labels/                        # Training labels
│   └── prepared/                      # Prepared datasets
├── docs/
│   ├── v4_PAT_ARCHITECTURE.md         # 🆕 v4 documentation
│   └── ...
├── mcp_servers/                       # MCP Servers
├── reports/                           # Training reports
├── MODEL_V3_REPORT.md                 # v3 final report
├── TECHNICAL_SPEC_v4_FUSION_TRANSFORMER.md  # 🆕 v4 spec
└── README.md

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

## 🎓 Training Pipeline v1.1

### 🚀 Автоматический конвейер обучения (рекомендуется)

**Один скрипт для полного цикла:**
```bash
python -m tools.train_and_backtest_hybrid \
    --symbol XAUUSD \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --primary-tf M5 \
    --seq-len 50 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3
```

**Этот скрипт выполнит:**
1. ✅ Экспорт данных из MT5 (если нужно)
2. ✅ Генерация меток через HybridStrategy бэктест
3. ✅ Подготовка датасета с фичами (11 признаков)
4. ✅ Обучение Direction LSTM с early stopping
5. ✅ Генерация отчёта в `reports/`

**Опции:**
- `--skip-export` - пропустить экспорт данных
- `--skip-training` - пропустить обучение, использовать текущую модель
- `--timeframes M1 M5 M15 H1 H4` - выбор таймфреймов

**Результат:**
- Модель: `models/direction_lstm_hybrid_{symbol}.pt`
- Метаданные: `models/direction_lstm_hybrid_{symbol}.json`
- Датасет: `data/prepared/direction_dataset_{symbol}.npz`
- Отчёт: `reports/hybrid_v1.1_{symbol}_{timestamp}.md`

### 🔧 Ручное обучение (для кастомизации)

#### Шаг 1: Экспорт данных из MT5
```bash
python -m tools.export_mt5_history \
    --symbol XAUUSD \
    --start 2024-01-01 \
    --end 2024-06-01 \
    --timeframes M1 M5 M15 H1 H4 \
    --format csv
```
- **Источник**: MetaTrader 5 через `get_ohlcv()`
- **Выход**: `data/raw/{symbol}/{timeframe}.csv`

#### Шаг 2: Генерация меток
```bash
python -m aimodule.training.generate_labels \
    --symbol XAUUSD \
    --primary-tf M5 \
    --data-dir data/raw \
    --output data/labels/direction_labels.csv
```
- **Метод**: Запуск HybridStrategy в бэктесте
- **Метки**: direction_label (0=FLAT, 1=LONG, 2=SHORT)
- **Выход**: `data/labels/direction_labels.csv` с трейдами

#### Шаг 3: Подготовка датасета
```bash
python -m aimodule.training.prepare_direction_dataset \
    --labels data/labels/direction_labels.csv \
    --data-dir data/raw \
    --symbol XAUUSD \
    --timeframe M5 \
    --seq-len 50 \
    --output data/prepared/direction_dataset.npz
```
- **Фичи**: 11 признаков (returns, SMA fast/slow/ratio, ATR/normalized, RSI, BB position, volume_ratio)
- **Нормализация**: StandardScaler (per-feature)
- **Последовательности**: sliding window seq_len × n_features
- **Сплит**: Train/Val/Test (80/20, then 80/20, stratified)
- **Выход**: `.npz` файл с X_train, y_train, X_val, y_val, X_test, y_test

#### Шаг 4: Обучение LSTM
```bash
python -m aimodule.training.train_direction_lstm_from_labels \
    --data data/prepared/direction_dataset.npz \
    --seq-len 50 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --save-path models/direction_lstm_hybrid.pt
```
- **Архитектура**: LSTM (2 слоя, 64 hidden units, dropout 0.3)
- **Метрики**: Accuracy, F1 macro, MCC (Matthews Correlation Coefficient)
- **Early stopping**: patience=5 epochs на validation MCC
- **Device**: CUDA если доступен (RTX 3070 ready)
- **Seed**: 42 для воспроизводимости
- **Выход**: `.pt` модель + `.json` метаданные

### 🧠 Legacy Training (старые модели)

#### Обучение Regime Model (KMeans/GMM)
```bash
python -m aimodule.training.train_regime_model
```
- **Метод**: KMeans или GaussianMixture
- **Признаки**: returns, ATR, SMA slope, volatility
- **Выход**: `models/regime_ml.pkl`

#### Обучение Direction Model (базовая LSTM)
```bash
python -m aimodule.training.train_direction_model
```
- **Архитектура**: LSTM (2 слоя, 64 hidden units)
- **Окно**: 100 свечей
- **Выход**: `models/direction_lstm.pt`

### 📊 Проверка результатов

После обучения проверьте:
```bash
# 1. Метаданные модели
cat models/direction_lstm_hybrid_{symbol}.json

# 2. Отчёт обучения
cat reports/hybrid_v1.1_{symbol}_{timestamp}.md

# 3. Запустите бэктест с новой моделью
python demo_backtest_hybrid.py
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

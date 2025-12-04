# 🎯 GOLD FEATURES INTEGRATION ROADMAP

## ✅ COMPLETED (03.12.2025 05:20)

### 1. Анализ Репозитория
- ✅ Изучено **15+ файлов** с LSTM реализациями для XAUUSD
- ✅ Найдено **4 Gold-специфических паттерна**:
  - Alpha Trend (RSI + ATR)
  - ICT Order Blocks + Liquidity Grab
  - Triple EMA + 200 EMA Filter
  - Multi-Timeframe Support/Resistance

### 2. Создание Модулей
- ✅ **`features_gold.py`** — модуль с 4 функциями:
  ```
  add_alpha_trend()                  # Волатильность + Momentum
  add_ict_order_blocks()             # Smart Money Concepts
  add_ema_institutional_filter()     # Институциональный фильтр
  add_support_resistance_static()    # Multi-TF уровни
  add_all_gold_features()            # Все сразу
  ```

- ✅ **Обновлен `features.py`**:
  - Добавлен параметр `use_gold_features=True`
  - Интегрирован импорт `features_gold`
  - Добавлена проверка минимума свечей (200+)

### 3. Тестирование
- ✅ Синтетические данные: **500 свечей M5**
- ✅ Результат: **17 новых фичей** добавлено корректно
- ✅ Все индикаторы считаются без ошибок

---

## 📋 NEXT STEPS (Порядок Выполнения)

### 🔥 STEP 1: Обновить Training Pipeline (Приоритет: КРИТИЧНО)

**Цель:** Включить новые фичи в обучение модели

**Файл:** `aimodule/learning/train_direction.py`

**Действия:**
```python
# Найти строку с FEATURE_COLUMNS
# Добавить новые Gold фичи:

from aimodule.data_pipeline.features_gold import GOLD_FEATURE_COLUMNS

FEATURE_COLUMNS = [
    'close', 'high', 'low', 'open', 'volume',
    'sma_fast', 'sma_slow', 'atr',
    
    # SMC Features
    'fvg_bullish', 'fvg_bearish', 'swing_high', 'swing_low',
    
    # НОВЫЕ: Gold Features
    *GOLD_FEATURE_COLUMNS  # Распаковка списка фичей
]
```

**Проверка:**
```bash
python -c "from aimodule.data_pipeline.features_gold import GOLD_FEATURE_COLUMNS; print(GOLD_FEATURE_COLUMNS)"
```

---

### 🔥 STEP 2: Экспорт Данных с Новыми Фичами (Приоритет: ВЫСОКИЙ)

**Цель:** Получить CSV с новыми колонками для обучения

**Команда:**
```bash
python tools/export_mt5_history.py XAUUSD M5 5000
```

**Ожидаемый результат:**
- Файл: `data/prepared/XAUUSD_M5_5000.csv`
- Размер: ~5000 строк × 30+ колонок (старые + новые)
- Новые колонки: AlphaTrend_*, Bullish_OB, EMA_20, EMA_50, EMA_200, etc.

**Проверка:**
```bash
python -c "import pandas as pd; df = pd.read_csv('data/prepared/XAUUSD_M5_5000.csv'); print(df.columns.tolist())"
```

---

### 🔥 STEP 3: Ретрейн Модели (Приоритет: КРИТИЧНО)

**Цель:** Обучить LSTM с новыми Gold фичами

**Команда:**
```bash
python tools/train_and_backtest_hybrid.py
```

**Настройки:**
- Epochs: 50 (как обычно)
- Features: Включить все GOLD_FEATURE_COLUMNS
- Save to: `models/direction_lstm_hybrid_XAUUSD_v2.pt`

**Ожидаемая метрика:**
- Val Accuracy: 60%+ (было ~55%)
- Val Loss: < 0.55 (было ~0.60)

**Проверка обучения:**
```bash
# После обучения проверяем файлы модели:
ls models/direction_lstm_hybrid_XAUUSD_v2.*
# Должно быть: .pt (веса) + .json (метаданные)
```

---

### 🔥 STEP 4: Бэктест с Новой Моделью (Приоритет: ВЫСОКИЙ)

**Цель:** Проверить улучшения на историческом периоде

**Команда:**
```bash
python demo_backtest_hybrid.py
```

**Изменения в коде:**
```python
# В demo_backtest_hybrid.py изменить путь к модели:
config = StrategyConfig(
    symbol='XAUUSD',
    primary_tf='M5',
    ai_direction_model="models/direction_lstm_hybrid_XAUUSD_v2.pt",  # Новая модель
    # ...
)
```

**Целевые метрики:**
| Метрика | Было | Цель | Прогноз |
|---------|------|------|---------|
| Win Rate | 46.15% | 55%+ | Alpha Trend + ICT OB |
| ROI | 100.02% | 120%+ | EMA Filter убирает bad setups |
| Max Drawdown | ? | -15% | Order Blocks ловят развороты |
| Trades Count | 13 | 20+ | Date parsing fix уже сработал |

---

### 📊 STEP 5: Сравнение Моделей (A/B Test)

**Цель:** Убедиться что новая модель лучше старой

**Метод:**
1. Запускаем бэктест на **одних и тех же данных**:
   - Model V1: `direction_lstm_hybrid_XAUUSD.pt` (старая)
   - Model V2: `direction_lstm_hybrid_XAUUSD_v2.pt` (с Gold фичами)

2. Сравниваем метрики в таблице:
   ```
   | Метрика        | Model V1 | Model V2 | Δ      |
   |----------------|----------|----------|--------|
   | Win Rate       | 46.15%   | ??       | +??    |
   | ROI            | 100.02%  | ??       | +??    |
   | Sharpe         | ?        | ??       | +??    |
   | Max Drawdown   | ?        | ??       | +??    |
   ```

3. Если Model V2 лучше → **деплоим на прод**
4. Если хуже → **анализируем причины** и дотюниваем гиперпараметры

---

## 🔬 OPTIONAL: Advanced Features (Для Будущего)

### 1. Multi-Timeframe Alpha Trend

**Идея:** Добавить Alpha Trend с разных таймфреймов (как в `newALV.py`)

**Реализация:**
```python
# В features_gold.py добавить:
def add_multi_tf_alpha_trend(df_m5, df_m15, df_h1, df_h4):
    """
    Alpha Trend с 4 таймфреймов
    Returns: 4 колонки - AlphaTrend_M5, AlphaTrend_M15, AlphaTrend_H1, AlphaTrend_H4
    """
    # Resample M5 → M15, H1, H4
    # Считаем Alpha Trend для каждого
    # Merge обратно в M5 dataframe
    pass
```

**Применение в Timeframe Selector:**
```python
# strategy/timeframe_selector.py
def scan_best_timeframe(self, symbol, ai_client):
    # NEW: Используем Multi-TF Alpha Trend для scoring
    alpha_scores = {
        'M5': df['AlphaTrend_M5'].iloc[-1],
        'M15': df['AlphaTrend_M15'].iloc[-1],
        'H1': df['AlphaTrend_H1'].iloc[-1],
        'H4': df['AlphaTrend_H4'].iloc[-1]
    }
    # Выбираем TF с max Alpha Trend Signal
    best_tf = max(alpha_scores, key=alpha_scores.get)
```

---

### 2. Dual-Output Model (Price + Risk)

**Идея:** Модель предсказывает ДВЕ вещи одновременно:
- Output 1: Future Price (регрессия)
- Output 2: Risk Label: Buy/Sell/Hold (классификация)

**Архитектура (из `newALV.py`):**
```python
# В aimodule/learning/train_direction.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

inputs = Input(shape=(seq_length, num_features))
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)

# Output 1: Price Prediction
price_output = Dense(1, name='price_output')(x)

# Output 2: Risk Management (Buy/Sell/Hold)
risk_output = Dense(3, activation='softmax', name='risk_output')(x)

model = Model(inputs=inputs, outputs=[price_output, risk_output])
model.compile(
    optimizer='adam',
    loss={'price_output': 'mse', 'risk_output': 'sparse_categorical_crossentropy'},
    loss_weights={'price_output': 0.7, 'risk_output': 0.3},
    metrics={'risk_output': 'accuracy'}
)
```

**Применение в HybridStrategy:**
```python
# strategy/hybrid_strategy.py
price_pred, risk_pred = ai_client.predict_dual(symbol, timeframe)
# price_pred: 2650.50
# risk_pred: [0.1, 0.8, 0.1] → BUY (80% confidence)

if risk_pred[1] > 0.85:  # High confidence BUY
    return self._generate_signal("buy", reason="AI Dual-Output High Conf BUY")
```

---

### 3. Temporal Cross-Validation

**Идея:** Вместо обычного train_test_split использовать TimeSeriesSplit

**Реализация:**
```python
# В aimodule/learning/train_direction.py
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold+1}/5")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    score = model.evaluate(X_val, y_val)
    fold_scores.append(score)

print(f"Average Val Accuracy: {np.mean(fold_scores):.4f}")
```

**Преимущество:**
- Лучше чем random split для временных рядов
- Избегаем data leakage (когда тестовые данные "из будущего")

---

## 📊 Success Criteria (Критерии Успеха)

### ✅ Minimum Viable Success (MVP)
- [ ] Win Rate ≥ 52% (было 46.15%)
- [ ] ROI ≥ 110% (было 100.02%)
- [ ] Trades Count ≥ 15 (было 13)

### 🏆 Target Success
- [ ] Win Rate ≥ 55%
- [ ] ROI ≥ 120%
- [ ] Sharpe Ratio ≥ 1.5
- [ ] Max Drawdown ≤ 15%

### 🚀 Exceptional Success
- [ ] Win Rate ≥ 60%
- [ ] ROI ≥ 150%
- [ ] Sharpe Ratio ≥ 2.0
- [ ] Consistency: Win Rate стабильна на разных периодах

---

## ⚠️ Риски и Митигация

### Риск 1: Переобучение на новых фичах
**Симптомы:**
- Train Accuracy: 95%+
- Val Accuracy: 45%
- Огромный разрыв между train/val

**Митигация:**
- Увеличить Dropout до 0.3
- Добавить L2 regularization
- Уменьшить количество LSTM layers

### Риск 2: NaN в новых фичах
**Симптомы:**
- Ошибка при обучении: "NaN loss detected"
- Проблема с EMA_200 на коротких данных

**Митигация:**
- Проверка: `df.isna().sum()` перед обучением
- Fillna стратегия: forward fill или drop
- Минимальная длина данных: 300 свечей

### Риск 3: Новые фичи не улучшают модель
**Симптомы:**
- Model V2 хуже Model V1 на бэктесте
- Win Rate падает вместо роста

**Митигация:**
- Feature Selection: убираем слабые фичи
- Hyperparameter Tuning: меняем архитектуру LSTM
- Ablation Study: тестируем фичи по отдельности

---

## 🗓️ Timeline (Временные Рамки)

### Сегодня (03.12.2025):
- ✅ 05:00-05:20 — Анализ репозитория + создание модулей
- ⏳ 05:30-06:00 — STEP 1: Обновить training pipeline
- ⏳ 06:00-07:00 — STEP 2: Экспорт данных + STEP 3: Ретрейн модели
- ⏳ 07:00-08:00 — STEP 4: Бэктест + анализ результатов

### Завтра (04.12.2025):
- STEP 5: A/B тестирование старой vs новой модели
- Если успех → деплой на demo account (live test)
- Если провал → root cause analysis

### На Неделю:
- Исследование Multi-TF Alpha Trend
- Dual-Output Model (Price + Risk)
- Temporal Cross-Validation integration

---

## 📚 Документация и Файлы

### Созданные Файлы:
1. ✅ **`XAUUSD_FEATURE_ANALYSIS.md`** — подробный анализ репозитория
2. ✅ **`features_gold.py`** — модуль с Gold фичами
3. ✅ **`GOLD_FEATURES_INTEGRATION_ROADMAP.md`** — этот файл (план действий)

### Обновленные Файлы:
1. ✅ **`features.py`** — добавлен импорт и параметр `use_gold_features`

### Требуют Обновления:
1. ⏳ **`train_direction.py`** — добавить GOLD_FEATURE_COLUMNS
2. ⏳ **`demo_backtest_hybrid.py`** — изменить путь к модели v2
3. ⏳ **`export_mt5_history.py`** — проверить что новые фичи экспортируются

---

## 🎯 Summary

**Текущий Статус:**
- ✅ Анализ завершен
- ✅ Код написан и протестирован
- ⏳ Готов к интеграции в pipeline

**Следующий Шаг:**
```bash
# 1. Обновляем train_direction.py
# 2. Экспортируем данные:
python tools/export_mt5_history.py XAUUSD M5 5000

# 3. Обучаем новую модель:
python tools/train_and_backtest_hybrid.py

# 4. Тестируем:
python demo_backtest_hybrid.py
```

**Ожидаемый Результат:**
- Win Rate: 46% → **55%+** (прирост +9%)
- ROI: 100% → **120%+** (прирост +20%)
- Trades: 13 → **20+** (больше качественных сетапов)

**ETA до Production:**
- MVP: **24 часа** (сегодня + завтра)
- Full Integration: **3-5 дней** (с A/B тестом и валидацией)

---

**Статус:** ✅ READY TO PROCEED  
**Next Action:** STEP 1 — Update `train_direction.py` with GOLD_FEATURE_COLUMNS

# 🌐 Мультитаймфреймовая Архитектура - Сводка Изменений

## 📅 Дата обновления: 2024-12-01

---

## 🎯 Цель

Интеграция мультитаймфреймовой логики в Golden Breeze Hybrid Strategy v1.0:

> **"Стратегия всегда читает все таймфреймы для принятия решений"**

---

## ✅ Реализованные изменения

### 1. 🆕 Новый модуль: `TimeframeSelector`

**Файл:** `strategy/timeframe_selector.py` (~350 строк)

**Функционал:**
- Динамический выбор `PRIMARY_TF` на основе AI сигналов по всем таймфреймам
- Анализ контекста старших таймфреймов (H1/H4)
- Фильтрация шума и волатильности
- История решений для отладки

**Классы:**
- `Timeframe(Enum)`: M1, M5, M15, H1, H4
- `Regime(Enum)`: TREND_UP, TREND_DOWN, RANGE, VOLATILE, UNKNOWN
- `TimeframeData(@dataclass)`: Данные по конкретному TF
- `TimeframeDecision(@dataclass)`: Решение селектора
- `TimeframeSelector`: Главный класс выбора TF

**Алгоритм выбора:**
1. Проверка M5: confidence ≥ 0.65, режим не volatile
2. Fallback на M15: если M5 недостаточно уверен
3. Fallback на H1: при высокой confidence на старшем TF
4. Контекстные фильтры: волатильность на H1/H4 повышает требования

---

### 2. 🔧 Обновлённый модуль: `AIClient`

**Файл:** `strategy/ai_client.py`

**Новые методы:**
- `predict_multitimeframe()` - запрос сигналов по всем TF одновременно
  - Payload: `{symbol, timeframes_data: {tf: [candles]}}`
  - Endpoint: `/predict_multitimeframe` (с fallback на `/predict` для каждого TF)
- `get_last_multitf_signals()` - получение последних мультитаймфреймовых сигналов

**Fallback механизм:**
- Если `/predict_multitimeframe` недоступен → последовательные запросы `/predict` для каждого TF

---

### 3. 📝 Обновлённый модуль: `StrategyConfig`

**Файл:** `strategy/config.py`

**Новые параметры:**

```python
# Мультитаймфреймовая логика
primary_tf: str = "M5"  # Динамически изменяемый рабочий TF
supported_timeframes: List[str] = ["M1", "M5", "M15", "H1", "H4"]
execution_tf: str = "M1"  # Младший TF для интрабара
context_tf_high: str = "H1"  # Старший контекстный TF

# Настройки TimeframeSelector
tf_selector_min_confidence: float = 0.65  # Мин confidence
tf_selector_high_confidence: float = 0.8  # Высокая confidence (фильтр волатильности)
tf_selector_enable: bool = True  # Включить динамический выбор

# Deprecated (для обратной совместимости)
base_timeframe: str = "M5"
intrabar_timeframe: str = "M1"
```

**Новая валидация:**
- Проверка `primary_tf` в `supported_timeframes`
- Проверка `execution_tf` и `context_tf_high`
- Проверка порогов confidence (0..1)

---

### 4. 🔄 Обновлённый модуль: `HybridStrategy`

**Файл:** `strategy/hybrid_strategy.py`

**Архитектурные изменения:**

1. **Инициализация:**
   - Добавлен `self.tf_selector: TimeframeSelector` (опционально)
   - Добавлен `self.current_primary_tf: str` (динамический)
   - Добавлен `self.current_multitf_signals: Dict[str, Dict]`
   - Добавлен `self.current_tf_decision: TimeframeDecision`
   - Добавлен `self.multitf_data: Dict[str, pd.DataFrame]`

2. **Метод `on_new_candle()` (переработан):**
   ```python
   def on_new_candle(
       candle: Dict, 
       historical_data: pd.DataFrame,
       multitf_data: Optional[Dict[str, pd.DataFrame]] = None  # НОВЫЙ параметр
   )
   ```

   **Новый workflow:**
   - ШАГ 1: `_request_multitf_signals()` → AI сигналы по M5/M15/H1/H4
   - ШАГ 2: `_select_primary_timeframe()` → выбор PRIMARY_TF через TimeframeSelector
   - ШАГ 3: Проверка `tf_decision.should_trade`
   - ШАГ 4: Генерация сигнала на PRIMARY_TF с данными из `multitf_data[PRIMARY_TF]`

3. **Новые методы:**
   - `_request_multitf_signals()` - запрос AI по всем TF
   - `_select_primary_timeframe()` - вызов TimeframeSelector
   - `_request_ai_signal()` - deprecated, но оставлен для обратной совместимости

4. **Обновлённый метод `get_statistics()`:**
   - Добавлено `"current_primary_tf"`
   - Добавлено `"tf_decision"` (если доступно)

---

### 5. 🔧 Обновлённый модуль: `BacktestEngine`

**Файл:** `strategy/backtest_engine.py`

**Новые методы:**

1. **`load_multitf_data(data_dict: Dict[str, pd.DataFrame])`**
   - Загружает данные по нескольким таймфреймам
   - Автоматически заполняет `self.multitf_data`
   - Совместим с `load_m5_data()` (обратная совместимость)

2. **`_sync_multitf_data(current_timestamp, current_index)`**
   - Синхронизирует данные по всем TF на момент текущей свечи
   - Фильтрует `data[data.index <= current_timestamp]`
   - Возвращает `Dict[str, pd.DataFrame]`

**Изменения в `run()`:**
```python
# Было:
self.strategy.on_new_candle(candle, historical_data)

# Стало:
synced_multitf_data = self._sync_multitf_data(current_bar.name, i)
self.strategy.on_new_candle(candle, historical_data, synced_multitf_data)
```

---

### 6. 📚 Новая документация

**Файлы:**
- `MULTITIMEFRAME_SPECIFICATION.md` (~500 строк)
  - Полное описание мультитаймфреймовой логики
  - Архитектура компонентов
  - Примеры сценариев (3 детальных case)
  - Настройка параметров
  - Workflow диаграммы
  - Roadmap для v1.1+

**Обновлённые файлы:**
- `README.md` - добавлены упоминания о multitimeframe
- `strategy/__init__.py` - экспорт TimeframeSelector и связанных классов

---

## 📊 Статистика кода

### Новые файлы:
- `strategy/timeframe_selector.py`: **350 строк**

### Изменённые файлы:
- `strategy/ai_client.py`: **+80 строк** (predict_multitimeframe, fallback)
- `strategy/config.py`: **+20 строк** (multitf параметры, валидация)
- `strategy/hybrid_strategy.py`: **+120 строк** (multitf workflow, новые методы)
- `strategy/backtest_engine.py`: **+50 строк** (load_multitf_data, sync)
- `strategy/__init__.py`: **+5 строк** (экспорт TimeframeSelector)

### Документация:
- `MULTITIMEFRAME_SPECIFICATION.md`: **500 строк**
- `README.md`: **+10 строк**

**Итого:** ~1135 строк кода и документации

---

## 🔌 Требования к AI Core

Для полной поддержки мультитаймфрейма AI Core должен реализовать:

### Новый endpoint: `/predict_multitimeframe`

**Request:**
```json
{
  "symbol": "XAUUSD",
  "timeframes_data": {
    "M5": [{"timestamp": "...", "open": 2600.0, ...}, ...],
    "M15": [...],
    "H1": [...],
    "H4": [...]
  }
}
```

**Response:**
```json
{
  "M5": {
    "regime": "trend_up",
    "direction": "long",
    "direction_confidence": 0.75,
    "sentiment": 0.5,
    "action": "enter_long",
    "reasons": [...]
  },
  "M15": {...},
  "H1": {...},
  "H4": {...}
}
```

**Fallback:** Если endpoint не реализован, `AIClient.predict_multitimeframe()` делает последовательные запросы к `/predict` для каждого TF.

---

## 🎮 Пример использования

### Простой пример (без мультитаймфрейма):

```python
from strategy import StrategyConfig, HybridStrategy, BacktestEngine

config = StrategyConfig(tf_selector_enable=False)  # Отключаем селектор
strategy = HybridStrategy(config, initial_balance=10000.0)
backtest = BacktestEngine(strategy, config)

backtest.load_m5_data(df_m5)
backtest.run()
```

### Пример с мультитаймфреймом:

```python
from strategy import StrategyConfig, HybridStrategy, BacktestEngine

config = StrategyConfig(
    primary_tf="M5",
    tf_selector_enable=True,
    tf_selector_min_confidence=0.65,
    tf_selector_high_confidence=0.8
)

strategy = HybridStrategy(config, initial_balance=10000.0)
backtest = BacktestEngine(strategy, config)

# Загружаем данные по всем таймфреймам
multitf_data = {
    "M5": df_m5,
    "M15": df_m15,
    "H1": df_h1,
    "H4": df_h4
}
backtest.load_multitf_data(multitf_data)
backtest.load_m1_data(df_m1)  # Для интрабара

# Запуск
backtest.run()
```

---

## ✅ Тестирование

### Проверка импортов:

```python
from strategy import (
    TimeframeSelector,
    TimeframeData,
    Timeframe,
    Regime,
    TimeframeDecision
)

# Создание селектора
selector = TimeframeSelector(
    default_primary_tf=Timeframe.M5,
    min_confidence_threshold=0.65
)

# Подготовка данных
tf_data = {
    Timeframe.M5: TimeframeData(
        timeframe=Timeframe.M5,
        regime=Regime.TREND_UP,
        direction="long",
        direction_confidence=0.75
    ),
    # ... остальные TF
}

# Выбор PRIMARY_TF
decision = selector.select_timeframe(tf_data)
print(f"PRIMARY_TF: {decision.primary_tf}")
print(f"Reason: {decision.reason}")
print(f"Should trade: {decision.should_trade}")
```

---

## 🚀 Преимущества новой архитектуры

1. **Адаптивность:** PRIMARY_TF меняется в зависимости от рыночных условий
2. **Контекстная осведомлённость:** Старшие TF фильтруют ложные сигналы
3. **Точность входа:** Младшие TF дают точные точки при тренде на H1/H4
4. **Защита от хаоса:** Автоматическое отключение при волатильности
5. **Прозрачность:** Логирование причин выбора PRIMARY_TF
6. **Обратная совместимость:** Старый код работает без изменений

---

## ⚠️ Ограничения v1.0

1. **Простые правила:** Фиксированные пороги confidence, без ML
2. **Нет self-learning:** Селектор не адаптируется на основе результатов
3. **Фиксированный набор TF:** M1, M5, M15, H1, H4 (нет D1, W1)
4. **AI Core endpoint:** Требуется реализация `/predict_multitimeframe`

---

## 📅 Roadmap v1.1+

1. **Adaptive TimeframeSelector:**
   - Обучение на основе ROI/WinRate по каждому TF
   - Динамическое изменение порогов

2. **Multi-TF confirmation:**
   - Требование совпадения направлений на 2+ TF

3. **News filter integration:**
   - Временное отключение за 15 мин до/после новостей

4. **Extended TF support:**
   - D1, W1 для долгосрочного контекста

---

## 📝 Заключение

Мультитаймфреймовая архитектура успешно интегрирована в Golden Breeze Hybrid Strategy v1.0:

✅ **350+ строк** нового кода (TimeframeSelector)  
✅ **270+ строк** обновлений существующих модулей  
✅ **500+ строк** документации  
✅ **Полная обратная совместимость**  
✅ **Готово к тестированию**  

**Статус:** ✅ Реализовано и задокументировано

---

**Версия:** 1.0  
**Дата:** 2024-12-01  
**Автор:** Golden Breeze Team

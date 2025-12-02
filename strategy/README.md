# Golden Breeze Hybrid Strategy v1.0

Гибридная торговая стратегия с интрабарной логикой для XAUUSD, интегрированная с Golden Breeze AI Core.

## 🎯 Особенности

### ✅ Что реализовано:

1. **Интрабарная логика**
   - Работа с тиками MT5
   - Симуляция через M1 данные
   - Реальное исполнение SL/TP/Trailing внутри свечи

2. **Адаптивные режимы**
   - **Trend**: Breakout strategy с частичным TP и трейлингом
   - **Range**: Mean reversion от границ диапазона
   - **Volatile**: Защитный режим (опционально NO TRADE)

3. **AI интеграция**
   - Сигналы от Golden Breeze AI Core
   - Автоматический feedback для self-learning
   - Использование regime, direction, confidence, sentiment

4. **Строгий Risk Management**
   - Фиксированный риск на сделку
   - Дневные и общие лимиты просадки
   - Ограничение количества позиций
   - Торговые сессии

5. **Backtesting**
   - Поддержка тиков и M1 данных
   - Equity curve и детальная статистика
   - Анализ по режимам рынка

---

## 📁 Структура

```
strategy/
├── __init__.py                 # Экспорт классов
├── config.py                   # StrategyConfig
├── intrabar_engine.py          # IntrabarEngine, Tick, IntrabarCandle
├── regime_strategies.py        # TrendStrategy, RangeStrategy, VolatileStrategy
├── risk_manager.py             # RiskManager, Trade
├── ai_client.py                # AIClient для связи с AI Core
├── hybrid_strategy.py          # HybridStrategy (главный класс)
└── backtest_engine.py          # BacktestEngine
```

---

## 🚀 Быстрый старт

### 1. Backtesting

```python
from strategy import StrategyConfig, HybridStrategy
from strategy.backtest_engine import BacktestEngine
import pandas as pd

# Конфигурация
config = StrategyConfig(
    symbol="XAUUSD",
    base_timeframe="M5",
    risk_per_trade_pct=1.0,
    max_daily_loss_pct=3.0,
    max_total_dd_pct=10.0,
    min_direction_confidence=0.65,
    ai_api_url="http://127.0.0.1:5005"
)

# Создание стратегии
strategy = HybridStrategy(config, initial_balance=10000.0)

# Backtesting engine
backtest = BacktestEngine(strategy, config)

# Загрузка данных (M5 с индикаторами)
m5_data = pd.read_csv("xauusd_m5_with_indicators.csv", index_col=0, parse_dates=True)
backtest.load_m5_data(m5_data)

# Опционально: M1 для интрабарной симуляции
m1_data = pd.read_csv("xauusd_m1.csv", index_col=0, parse_dates=True)
backtest.load_m1_data(m1_data)

# Запуск
backtest.run(start_date="2024-01-01", end_date="2024-12-01")

# Экспорт результатов
backtest.export_results("backtest_results.csv")
```

### 2. Live Trading (с MT5)

```python
from strategy import StrategyConfig, HybridStrategy
from mcp_servers.trading import market_data, MT5Connector
import time

# Подключение к MT5
connector = MT5Connector()
connector.initialize()

# Конфигурация
config = StrategyConfig(
    symbol="XAUUSD",
    session_start_utc=2,
    session_end_utc=22
)

# Создание стратегии
strategy = HybridStrategy(config, initial_balance=10000.0)

# Основной цикл
while True:
    # Получение M5 данных
    df = market_data.get_ohlcv("XAUUSD", "M5", count=200)
    
    # Последняя свеча
    last_candle = {
        "timestamp": str(df.index[-1]),
        "open": df.iloc[-1]["open"],
        "high": df.iloc[-1]["high"],
        "low": df.iloc[-1]["low"],
        "close": df.iloc[-1]["close"],
        "volume": df.iloc[-1]["volume"]
    }
    
    # Обработка
    strategy.on_new_candle(last_candle, df)
    
    # Ожидание следующей свечи (5 минут)
    time.sleep(300)
```

---

## ⚙️ Конфигурация

### Основные параметры

```python
config = StrategyConfig(
    # Инструмент
    symbol="XAUUSD",
    base_timeframe="M5",
    intrabar_timeframe="M1",
    
    # Сессия (UTC)
    session_start_utc=2,   # 02:00
    session_end_utc=22,    # 22:00
    
    # Risk Management
    risk_per_trade_pct=1.0,        # 1% риска на сделку
    max_daily_loss_pct=3.0,        # -3% макс дневная просадка
    max_total_dd_pct=10.0,         # -10% макс общая просадка
    max_positions=3,               # Максимум 3 позиции одновременно
    max_bars_hold=100,             # Максимум 100 баров в позиции
    
    # AI Core
    ai_api_url="http://127.0.0.1:5005",
    min_direction_confidence=0.65,  # Мин confidence для входа
    min_sentiment_threshold=-0.2,   # Мин sentiment
    
    # Trend режим
    trend_partial_tp_pct=50.0,         # 50% частичный TP
    trend_trailing_atr_mult=2.0,       # Трейлинг = ATR * 2
    trend_min_profit_for_trail=0.5,    # Мин 0.5R для трейлинга
    
    # Range режим
    range_tp_fixed_points=100.0,       # Фикс TP = 100 пунктов
    range_max_atr_threshold=150.0,     # Макс ATR для range
    range_rsi_oversold=30.0,
    range_rsi_overbought=70.0,
    
    # Volatile режим
    volatile_risk_reduction=0.5,       # Риск * 0.5
    volatile_min_confidence=0.8,       # Высокий порог
    volatile_allow_trades=False        # По умолчанию NO TRADE
)
```

---

## 📊 Логика по режимам

### 🔥 Trend (trend_up / trend_down)

**Стиль:** Trend-following с пробоями

**Условия входа:**
- Пробой локального уровня (max/min за 20 баров)
- Direction совпадает с трендом
- Confidence ≥ 0.65
- Sentiment ≥ -0.2

**Управление позицией:**
- Частичный TP на первом уровне (50% позиции)
- Trailing stop после 0.5R прибыли
- Trailing distance = ATR × 2.0

### 〰️ Range (range)

**Стиль:** Mean reversion

**Условия входа:**
- Цена у границы диапазона (±1%)
- RSI < 30 (для long) или RSI > 70 (для short)
- ATR < 150 (низкая волатильность)
- Sentiment ≈ нейтральный

**Управление позицией:**
- Фиксированный TP = 100 пунктов
- SL за границей диапазона
- Max bars hold = 100

### ⚠️ Volatile (volatile)

**Стиль:** Защитный (по умолчанию NO TRADE)

**Условия входа (если allow_trades=True):**
- Confidence ≥ 0.8 (очень высокий)
- Sentiment ≥ -0.2

**Управление позицией:**
- Риск × 0.5 (уменьшение)
- Wider stops (SL × 1.5, TP × 1.5)

---

## 🎯 Risk Management

### Лимиты

- **Риск на сделку:** 1% от депозита (настраивается)
- **Дневная просадка:** Макс -3%
- **Общая просадка:** Макс -10%
- **Max positions:** 3 одновременные позиции
- **Max bars hold:** 100 баров (особенно для range)

### Торговые сессии

- **UTC 02:00–22:00** (настраивается)
- Вне сессии: новые позиции не открываются

### Position Sizing

```python
# Автоматический расчёт по риску
volume = risk_amount / (sl_distance * point_value)

# Корректировка для volatile режима
volume *= risk_reduction  # 0.5 для volatile
```

---

## 🔗 Интеграция с AI Core

### Запрос сигнала

Стратегия запрашивает AI сигнал **по закрытию M5 свечи**:

```python
{
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "candles": [...]  # Последние 100 свечей
}
```

### Ответ AI

```python
{
    "regime": "trend_up",
    "direction": "long",
    "direction_confidence": 0.75,
    "sentiment": 0.3,
    "action": "enter_long",
    "reasons": ["Strong uptrend", "Positive sentiment"]
}
```

### Feedback после сделки

```python
{
    "symbol": "XAUUSD",
    "regime": "trend_up",
    "direction": "long",
    "sentiment": 0.3,
    "result_pnl": 150.0,
    "good_trade": True
}
```

---

## 📈 Метрики

### Общая статистика

- Date Start / End
- Timeframe
- ROI (%)
- Net PnL ($)
- Win Ratio (%)
- Max Drawdown (%)
- Time in Market (%)
- Number of Trades
- Average Trade Duration

### По режимам

- PnL по каждому режиму (trend_up, trend_down, range, volatile)
- Win Rate по режимам
- Количество сделок по режимам

---

## 🧪 Тестирование

### Unit Tests

```bash
pytest tests/test_hybrid_strategy.py -v
```

### Backtesting на истории

```bash
python demo_backtest_hybrid.py
```

---

## 📝 TODO / Roadmap

### v1.0 (текущая)
- [x] Интрабарная логика (тики/M1)
- [x] 3 режима (trend/range/volatile)
- [x] Risk management
- [x] AI интеграция
- [x] Backtesting engine

### v1.1 (ближайшее будущее)
- [ ] Partial close для trend режима
- [ ] News filter integration
- [ ] Multi-timeframe confirmation
- [ ] Advanced trailing (ATR-based zones)

### v2.0 (среднесрочно)
- [ ] Level detection module (S/R zones)
- [ ] Volume profile analysis
- [ ] Order flow integration
- [ ] Machine learning для level detection

---

## 📞 Support

- Issues: GitHub Issues
- Docs: См. TZ_HYBRID_STRATEGY.md
- AI Core: См. README.md в корне проекта

---

**Версия:** v1.0.0  
**Дата:** 01 декабря 2025  
**Статус:** ✅ Production Ready (backtesting)

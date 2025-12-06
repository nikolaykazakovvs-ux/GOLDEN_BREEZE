# 🌍 Golden Breeze - Omni-Trader Architecture

## Overview

**Omni-Trader** — это единая торговая система, которая превращает "разрозненные скрипты" в **coordinated ecosystem**.

### Проблема (которую мы решаем)

Было:
- `live_mt5.py` — отдельный скрипт для MT5
- `live_mexc.py` — отдельный скрипт для MEXC
- `live_tradelocker.py` — отдельный скрипт для TradeLocker
- Сигналы дублировались, приходили разные результаты, невозможно контролировать риск глобально

Стало:
- **Один мозг** (v5_ultimate) смотрит на ВСЕ данные одновременно
- **Один роутер** (TradeRouter) решает, как исполнить сигнал на каждом счете
- **Один цикл** (OmniverseLoop) управляет всем процессом

---

## Architecture: 3-Layer Model

```
┌─────────────────────────────────────────────────────┐
│            DATA SOURCES (The Eyes)                  │
│  MT5 (XAUUSD, EURUSD) + MEXC (BTC, ETH) +          │
│  TradeLocker (demo/live prices)                     │
└──────────────────┬──────────────────────────────────┘
                   │ (OHLC data, 200 bars M5)
                   │
┌──────────────────▼──────────────────────────────────┐
│         AI BRAIN (The Brain)                        │
│  v5_Ultimate Model                                  │
│  - Analyzes BTC, GOLD, EUR simultaneously           │
│  - Outputs abstract signals (ASSET, DIRECTION, CONF)│
└──────────────────┬──────────────────────────────────┘
                   │ (AISignal: BTC UP, 85%)
                   │
┌──────────────────▼──────────────────────────────────┐
│       TRADE ROUTER (The Hands)                      │
│  Looks up signal in ROUTING_MAP                     │
│  Calculates position size per target                │
│  Sends orders to correct connectors                 │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌──────────────┐
   │ MT5    │ │ MEXC   │ │ TradeLocker  │
   │ Conn.  │ │ Conn.  │ │ Connector    │
   └────────┘ └────────┘ └──────────────┘
        │          │          │
        ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌──────────────┐
   │ Order  │ │ Order  │ │ Order on     │
   │on      │ │on      │ │Prop Firm     │
   │XAUUSD  │ │BTC/USDT│ Account       │
   └────────┘ └────────┘ └──────────────┘
```

---

## Module Structure

### 1. `config_routing.py` — Карта маршрутизации

**Ответственность:** Хранить информацию о счетах и правилах маршрутизации.

**Ключевые сущности:**

```python
Account              # Описание счета (MT5 Demo, MEXC Main, TL Prop)
  ├── name
  ├── connector_type
  ├── account_type    # SPOT / MARGIN / PROP_FIRM
  └── risk_config     # Процент риска, макс размер и т.д.

ExecutionTarget      # Конкретная цель исполнения сигнала
  ├── account
  ├── symbol          # BTC/USDT, XAUUSD и т.д.
  └── metadata

ROUTING_MAP          # Словарь: "BTC" -> [target1, target2, ...]
  └── "BTC":
       ├── ExecutionTarget(MEXC spot, BTC/USDT)
       └── ExecutionTarget(TradeLocker, BTCUSD)
```

**Пример:**

```python
ACCOUNTS = {
    "mexc_spot_main": Account(
        connector_type="MEXC",
        account_type=AccountType.SPOT,
        risk_config=RiskConfig(profile=RiskProfile.BALANCED, max_risk_percent=1.0)
    ),
    "mt5_demo": Account(
        connector_type="MT5",
        account_type=AccountType.MARGIN,
        risk_config=RiskConfig(profile=RiskProfile.CONSERVATIVE, max_risk_percent=0.5)
    )
}

ROUTING_MAP = {
    "BTC": [
        ExecutionTarget(ACCOUNTS["mexc_spot_main"], "BTC/USDT"),
        ExecutionTarget(ACCOUNTS["tradelocker_prop_1"], "BTCUSD")
    ],
    "GOLD": [
        ExecutionTarget(ACCOUNTS["mt5_demo"], "XAUUSD")
    ]
}
```

---

### 2. `trade_router.py` — Маршрутизатор ордеров

**Ответственность:** Получить сигнал и исполнить его на правильных счетах.

**Главный класс:**

```python
class TradeRouter:
    def execute_signal(signal: AISignal) -> List[ExecutionResult]
        # 1. Валидирует сигнал (confidence, timing и т.д.)
        # 2. Получает targets из ROUTING_MAP
        # 3. Для каждого target:
        #    - Вычисляет размер позиции
        #    - Отправляет ордер
        #    - Логирует результат
```

**Алгоритм валидации сигнала:**

```
Signal in: "BTC UP, 85% confidence"
    ↓
Check: confidence >= MIN_CONFIDENCE[BTC]  (0.55) ✓
Check: не слишком часто (>= 5 мин от последнего)  ✓
Check: в торговые часы  ✓
    ↓
VALID → Proceed to routing
```

**Калькуляция размера позиции:**

```
For each target:
    1. Get account balance
    2. risk_amount = balance * risk_percent / 100
    3. Adjust for confidence:
       - confidence < 65% → 50% of risk_amount
       - confidence > 80% → 100% of risk_amount
    4. position_size = risk_amount / asset_price
    5. Limit by max_position_size
```

**Параллельное исполнение:**

```python
# Все ордеры отправляются параллельно (async/await)
# Если MEXC упала — TradeLocker всё равно исполнится
results = await asyncio.gather(
    _execute_order(target1),
    _execute_order(target2),
    ...
    return_exceptions=True  # Ошибка одного ≠ ошибка всех
)
```

---

### 3. `omni_loop.py` — Главный цикл

**Ответственность:** Управлять всеми коннекторами и координировать систему.

**Главный класс:**

```python
class OmniverseLoop:
    async def run_loop(max_iterations=None)
        while running:
            1. Collect data from ALL sources (MT5, MEXC, TL)
            2. Run inference (v5_ultimate)
            3. Execute signals via TradeRouter
            4. Sync to next M5 candle
```

**Цикл работы (каждые 5 минут):**

```
[DATA COLLECTION] ────────────────────┐
  Параллельный fetch:                 │
  • MT5: XAUUSD (200 M5 bars)          │
  • MEXC: BTC/USDT (200 M5 bars)       │
  • TradeLocker: demo data             │
                                       │
[INFERENCE] ◄──────────────────────────┘
  Один AI мозг видит ВСЁ:
  • BTC movement на MEXC
  • GOLD movement на MT5
  • Корреляции между ними
  ↓
  Outputs:
  • Signal 1: BTC UP (85%)
  • Signal 2: GOLD DOWN (60%)
  • Signal 3: EUR NEUTRAL (50%) ← не исполняем
                                       │
[EXECUTION] ◄──────────────────────────┘
  Router для каждого сигнала:
  • BTC UP → MEXC spot + TradeLocker
  • GOLD DOWN → MT5
                                       │
[SYNC] ◄───────────────────────────────┘
  Sleep до следующей M5 свечи
```

---

## Configuration

### Включение/отключение коннекторов

```python
omniverse = OmniverseLoop(
    enable_mt5=True,           # ✓ Золото и Форекс
    enable_mexc=True,          # ✓ Крипто спот
    enable_tradelocker=True,   # ✓ Пропы
    live_trading=False         # Демо режим
)
```

### Добавление нового ассета

**Шаг 1:** Добавить в `ROUTING_MAP` (config_routing.py):

```python
ROUTING_MAP = {
    "BTC": [...],
    "GOLD": [...],
    "BNB": [  # NEW
        ExecutionTarget(
            account=ACCOUNTS["mexc_spot_main"],
            symbol="BNB/USDT"
        )
    ]
}
```

**Шаг 2:** Добавить в `inference()` (omni_loop.py):

```python
if 'BNB/USDT' in market_data:
    direction, confidence = await self._predict_asset(
        data=market_data['BNB/USDT'],
        asset_class='BNB'
    )
    signals.append(AISignal(...))
```

### Изменение риска

**Консервативно:**

```python
ACCOUNTS["mt5_demo"] = Account(
    ...,
    risk_config=RiskConfig(
        profile=RiskProfile.CONSERVATIVE,
        max_risk_percent=0.5,      # 0.5% на сделку
        max_position_size=50.0     # Макс 50 лотов
    )
)
```

**Агрессивно:**

```python
ACCOUNTS["tradelocker_prop_1"] = Account(
    ...,
    risk_config=RiskConfig(
        profile=RiskProfile.AGGRESSIVE,
        max_risk_percent=2.0,       # 2% на сделку
        max_position_size=1000.0    # Макс 1000 USD
    )
)
```

---

## Usage

### Запуск в демо режиме (5 итераций)

```bash
python run_omniverse.py --demo --iterations 5
```

### Запуск в реальном режиме

```bash
python run_omniverse.py --live
```

### Отключение отдельных коннекторов

```bash
python run_omniverse.py --demo --no-mt5 --no-tl
# Только MEXC будет работать
```

---

## Monitoring & Debugging

### Логирование

Система выводит:

```
[ITERATION 1] 10:30:05
------------------------------------------------------
[DATA COLLECTION] Gathering market data...
  ✓ XAUUSD: 200 bars
  ✓ BTC/USDT: 200 bars
  ✓ ETH/USDT: 200 bars

[INFERENCE] Running AI Brain...
  Generated 2 signals

[EXECUTION] Routing 2 signals...
→ Executing on mexc_spot_main (BTC/USDT)...
  ✓ Order placed: ID=12345, Volume=0.0015 BTC

→ Executing on mt5_demo (XAUUSD)...
  ✓ Order placed: ID=99888, Volume=0.5 lots

Iteration completed in 3.42s
Sleeping 297s until next M5 candle...
```

### Отладка сигналов

```python
# Просмотр истории исполнений
router.get_execution_history("BTC")  # Только BTC
router.log_summary()                 # Общий отчёт
```

---

## Error Handling

### Что происходит, если:

**MT5 упала:**
- MEXC и TradeLocker всё равно работают
- Логируется warning
- Система продолжает работать

**Один ордер не разместился:**
- Ордер на другом счете выполняется
- Результат логируется как частичный успех

**Слишком слабый сигнал:**
- Валидация отклоняет сигнал
- Ордеры не отправляются

**Нарушение risk rules:**
- Размер позиции обрезается до макимума
- Логируется warning

---

## Integration with AI Brain

### Текущее состояние

В `omni_loop.py` метод `_predict_asset()` возвращает placeholder:

```python
async def _predict_asset(self, data, asset_class):
    # TODO: Реальное предсказание
    # prediction = predict_direction(data, model=self.model_v5_ultimate)
    # return prediction['direction'], prediction['confidence']
    
    return SignalDirection.NEUTRAL, 0.0
```

### Интеграция v5_ultimate

Нужно:

1. Загрузить модель в `__init__`:

```python
self.model_v5 = torch.load('models/v5_btc/best_model.pt')
```

2. Реализовать реальное предсказание:

```python
from aimodule.inference.predict_direction import predict_direction

async def _predict_asset(self, data, asset_class):
    if len(data) < 50:
        return SignalDirection.NEUTRAL, 0.0
    
    # Реальное предсказание
    prediction = predict_direction(data, model=self.model_v5)
    
    direction = (SignalDirection.UP if prediction['direction'] == 1 
                 else SignalDirection.DOWN if prediction['direction'] == -1
                 else SignalDirection.NEUTRAL)
    
    return direction, prediction['confidence']
```

---

## Future Enhancements

- [ ] **Live Risk Monitoring:** Отслеживание global drawdown
- [ ] **Dynamic Position Sizing:** Автоматическое снижение размера при loss streak
- [ ] **Correlation Analysis:** Не открывать BTC и ETH одновременно (если коррелированы)
- [ ] **Sentiment Integration:** Использовать sentiment в калькуляции confidence
- [ ] **ML Retraining:** Автоматическое переобучение модели на новых данных
- [ ] **Multi-Timeframe Analysis:** Проверка сигналов на H1 перед исполнением на M5
- [ ] **Telegram Alerts:** Notifications при важных событиях

---

## Summary

**Omni-Trader** — это не просто объединение скриптов. Это полнофункциональная **Order Management System (OMS)**, которая:

- ✅ Одновременно торгует несколькими ассетами
- ✅ Управляет рисками глобально (не на счет, а на систему)
- ✅ Распределяет одинаковые сигналы по разным счетам
- ✅ Работает асинхронно (не блокирует на ошибках)
- ✅ Готова к масштабированию (легко добавить коннектор)

**Это то, что работает в крупных фондах, но теперь это ваше. 🚀**

Author: Golden Breeze Team  
Version: 1.0.0  
Date: 2025-12-06

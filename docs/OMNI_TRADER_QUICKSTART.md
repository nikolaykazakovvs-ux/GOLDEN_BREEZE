# 🚀 Omni-Trader Quick Start Guide

## TL;DR

Omni-Trader — это система управления ордерами, которая:

1. **Собирает данные** со всех источников одновременно
2. **Анализирует** единой моделью AI
3. **Маршрутизирует** сигналы на конкретные счета
4. **Исполняет** параллельно на всех счетах

---

## Быстрый старт

### 1. Запуск в демо режиме

```bash
python run_omniverse.py --demo --iterations 1
```

**Что произойдёт:**
- ✓ Подключится к MT5 (если доступен)
- ✓ Подключится к MEXC (если доступны API ключи)
- ✓ Собёрёт данные за последние 200 M5 свечей
- ✓ Запустит AI модель
- ✓ Выведет сигналы (но не будет их исполнять)

### 2. Запуск без отдельных коннекторов

```bash
# Только MEXC
python run_omniverse.py --demo --no-mt5 --no-tl

# Только MT5
python run_omniverse.py --demo --no-mexc --no-tl
```

### 3. Реальная торговля

```bash
python run_omniverse.py --live
```

⚠️ **Убедитесь, что все конфиги заполнены правильно!**

---

## Что где находится

| Файл | Назначение |
|------|-----------|
| `aimodule/manager/config_routing.py` | Карта счетов и маршрутизации |
| `aimodule/manager/trade_router.py` | Логика маршрутизации и исполнения |
| `aimodule/manager/omni_loop.py` | Главный цикл системы |
| `run_omniverse.py` | Entry point для запуска |
| `docs/OMNI_TRADER_ARCHITECTURE.md` | Полная документация |

---

## Как добавить новый счёт

### Пример: Добавить ещё один счет на MEXC

```python
# aimodule/manager/config_routing.py

ACCOUNTS["mexc_secondary"] = Account(
    name="mexc_secondary",
    connector_type="MEXC",
    account_type=AccountType.SPOT,
    enabled=True,
    risk_config=RiskConfig(
        profile=RiskProfile.CONSERVATIVE,
        max_risk_percent=0.5
    ),
    metadata={
        "api_key": None,  # ← Заполнить переменными окружения
        "api_secret": None
    }
)
```

---

## Как добавить новый ассет

### Пример: BNB

**Шаг 1:** Добавить в ROUTING_MAP

```python
# config_routing.py
ROUTING_MAP = {
    ...,
    "BNB": [
        ExecutionTarget(
            account=ACCOUNTS["mexc_spot_main"],
            symbol="BNB/USDT"
        )
    ]
}
```

**Шаг 2:** Добавить в inference

```python
# omni_loop.py, метод inference()

if 'BNB/USDT' in market_data:
    direction, confidence = await self._predict_asset(
        data=market_data['BNB/USDT'],
        asset_class='BNB'
    )
    if direction != SignalDirection.NEUTRAL:
        signals.append(AISignal(...))
```

**Шаг 3:** Добавить в сбор данных

```python
# omni_loop.py, метод collect_market_data()

if 'MEXC' in self.connectors:
    for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:  # ← NEW
        task = self._fetch_symbol_data('MEXC', symbol, 'M5', 200)
        tasks.append((symbol, task))
```

---

## Как изменить риск

### Консервативная торговля

```python
# config_routing.py
risk_config=RiskConfig(
    profile=RiskProfile.CONSERVATIVE,
    max_risk_percent=0.5,      # 0.5% на сделку
    max_position_size=100.0    # Макс размер
)
```

### Агрессивная торговля

```python
risk_config=RiskConfig(
    profile=RiskProfile.AGGRESSIVE,
    max_risk_percent=2.0,      # 2% на сделку
    max_position_size=5000.0   # Большой размер
)
```

---

## Понимание сигналов

### Что такое AISignal

```python
AISignal(
    asset_class='BTC',           # Какой ассет
    direction=SignalDirection.UP, # Куда идти (UP/DOWN/NEUTRAL)
    confidence=0.85,             # Уверенность (0-1)
    timestamp=datetime.now()
)
```

### Валидация сигнала

Сигнал НЕ исполняется, если:

- ❌ `confidence < MIN_CONFIDENCE[asset]` (для BTC это 0.55)
- ❌ Слишком часто от последнего сигнала (< 5 мин)
- ❌ Вне торговых часов (для EUR это 08:00-22:00 UTC)
- ❌ `direction == NEUTRAL`

---

## Мониторинг

### Просмотр логов

```bash
# Запуск с максимальным логированием
python run_omniverse.py --demo 2>&1 | tee omniverse.log

# Просмотр только ошибок
grep "✗" omniverse.log
```

### Статистика

После завершения система выводит:

```
OMNIVERSE STATISTICS
======================================================================
Uptime: 0:30:15
Loop Iterations: 6
Signals Processed: 12
Orders Executed: 8
Errors: 1
```

### Историю исполнений

```python
# В коде можно получить историю
results = router.get_execution_history("BTC")
for result in results:
    print(f"{result.signal.asset_class}: {result.success}")
```

---

## Troubleshooting

### MT5 не подключается

```
ERROR: MT5 connection failed
```

**Решение:**
1. Проверьте, что MT5 запущен
2. Заполните credentials в `config.py`
3. Запустите с `--no-mt5` для временного отключения

### MEXC API ошибка

```
ERROR: MEXC connection failed: Invalid API key
```

**Решение:**
1. Проверьте API ключи в переменных окружения
2. Убедитесь, что ключ имеет права на trading
3. Запустите с `--no-mexc` для отключения

### Сигналы не исполняются

**Проверьте:**
1. `confidence` выше минимального порога (см. `SIGNAL_FILTER_RULES`)
2. Ассет есть в `ROUTING_MAP`
3. Счет включён (`enabled=True`)
4. Баланс счета достаточен

---

## Examples

### Пример 1: Только демо на MEXC

```bash
python run_omniverse.py --demo --no-mt5 --no-tl --iterations 3
```

### Пример 2: Мониторинг с сохранением логов

```bash
python run_omniverse.py --demo 2>&1 | tee logs/omniverse_$(date +%Y%m%d_%H%M%S).log
```

### Пример 3: Прямой импорт в Python скрипте

```python
import asyncio
from aimodule.manager import OmniverseLoop

async def main():
    omniverse = OmniverseLoop(
        enable_mt5=True,
        enable_mexc=True,
        enable_tradelocker=False,
        live_trading=False
    )
    await omniverse.run_loop(max_iterations=10)

asyncio.run(main())
```

---

## Next Steps

1. ✅ Убедитесь, что все коннекторы работают в демо режиме
2. ✅ Проверьте, что сигналы генерируются корректно
3. ✅ Тестируйте на демо счетах несколько дней
4. ⚠️ Потом переходите на `--live`

---

## Support

- 📖 Полная документация: `docs/OMNI_TRADER_ARCHITECTURE.md`
- 🔧 Коды ошибок: см. логи системы
- 💬 Вопросы: смотрите комментарии в коде

---

**Omni-Trader v1.0.0** — это ваш путь к профессиональной торговле! 🚀

Author: Golden Breeze Team  
Date: 2025-12-06

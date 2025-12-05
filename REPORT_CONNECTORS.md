# 🔌 MULTI-CONNECTOR SYSTEM - REPORT

**Дата:** 2025-12-06
**Версия:** 1.0.0
**Статус:** ✅ ГОТОВО

---

## 📋 SUMMARY

Создана унифицированная система коннекторов для работы с разными торговыми платформами через единый интерфейс.

### Поддерживаемые платформы:

| Платформа | Тип | Статус | Описание |
|-----------|-----|--------|----------|
| **MT5** | Forex/CFD | ✅ Ready | MetaTrader 5 через Python API |
| **MEXC** | Crypto Spot/Futures | ✅ Tested | Криптобиржа через ccxt |
| **TradeLocker** | Prop Firms | ✅ Ready | REST API с Token Auth |

---

## 🏗️ ARCHITECTURE

```
aimodule/connector/
├── __init__.py          # Экспорт всех классов
├── base.py              # Базовые классы и типы
├── mt5.py               # MT5 коннектор
├── mexc.py              # MEXC коннектор
└── tradelocker.py       # TradeLocker коннектор

aimodule/data_pipeline/
└── data_manager.py      # Унифицированный менеджер данных
```

---

## 📦 CREATED FILES

### 1. `aimodule/connector/base.py`
Базовые классы:
- `OrderSide` - enum (BUY, SELL)
- `OrderType` - enum (MARKET, LIMIT, STOP, STOP_LIMIT)
- `OrderResult` - dataclass результата ордера
- `Position` - dataclass открытой позиции
- `AccountInfo` - dataclass информации об аккаунте
- `BaseConnector` - абстрактный класс с интерфейсом

### 2. `aimodule/connector/mt5.py`
MetaTrader 5 коннектор:
- Подключение через MT5 Python API
- Все стандартные операции (history, balance, orders, positions)
- Маппинг таймфреймов MT5

### 3. `aimodule/connector/mexc.py`
MEXC коннектор:
- Использует ccxt library
- Поддержка spot и futures
- Публичные данные без API ключей
- Торговля с API ключами

### 4. `aimodule/connector/tradelocker.py`
TradeLocker коннектор:
- REST API с JWT авторизацией
- Автоматическое обновление токенов
- Поддержка prop-фирм

### 5. `aimodule/data_pipeline/data_manager.py`
Унифицированный менеджер:
- `fetch_data()` - получение из любого источника
- `save_data()` - сохранение в parquet
- `load_data()` - загрузка сохранённых данных
- `fetch_training_data()` - сбор данных для обучения

### 6. `aimodule/config.py` (обновлён)
Добавлены секции:
- `MT5_CONFIG` - настройки MT5
- `MEXC_CONFIG` - настройки MEXC
- `TRADELOCKER_CONFIG` - настройки TradeLocker
- `DEFAULT_DATA_SOURCE` - источник по умолчанию
- `SOURCE_SYMBOLS` - символы для каждого источника

---

## 🧪 TEST RESULTS

```
MEXC Connector Test:
✅ Подключение: 3323 рынков
✅ Текущая цена BTC: $89,266.49
✅ История: 100 баров BTC/USDT 1h
✅ Сохранение: data/raw/mexc/BTC_USDT/1h.parquet
```

---

## 🚀 USAGE EXAMPLES

### Быстрый старт
```python
from aimodule.data_pipeline.data_manager import DataManager

# Создаём менеджер
dm = DataManager()

# Получаем данные из MEXC
df_crypto = dm.fetch_data(
    source="mexc",
    symbol="BTC/USDT",
    timeframe="1h",
    count=1000
)

# Получаем данные из MT5
df_forex = dm.fetch_data(
    source="mt5",
    symbol="XAUUSD",
    timeframe="H1",
    count=1000
)

# Получаем данные для обучения
training_data = dm.fetch_training_data(
    source="mexc",
    symbol="ETH/USDT",
    timeframes=["15m", "1h", "4h"],
    days_back=365
)
```

### Прямое использование коннектора
```python
from aimodule.connector import MEXCConnector, OrderSide, OrderType

# Подключаемся с API ключами для торговли
connector = MEXCConnector(
    api_key="your_api_key",
    api_secret="your_secret"
)
connector.connect()

# Получаем баланс
balance = connector.get_balance()
print(f"USDT: ${balance}")

# Размещаем ордер
result = connector.place_order(
    symbol="BTC/USDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    volume=0.001
)

if result.success:
    print(f"Order ID: {result.order_id}")
```

---

## 📁 DATA STORAGE

Данные сохраняются в структуре:
```
data/raw/
├── mexc/
│   ├── BTC_USDT/
│   │   ├── 15m.parquet
│   │   ├── 1h.parquet
│   │   └── metadata.json
│   └── ETH_USDT/
│       └── ...
├── mt5/
│   └── XAUUSD/
│       ├── M15.parquet
│       ├── H1.parquet
│       └── metadata.json
└── tradelocker/
    └── ...
```

---

## ⚙️ CONFIGURATION

Добавьте в `aimodule/config.py`:

```python
# MEXC
MEXC_CONFIG = {
    "api_key": "your_key",
    "api_secret": "your_secret",
    "testnet": False,
    "market_type": "spot"
}

# TradeLocker
TRADELOCKER_CONFIG = {
    "email": "your@email.com",
    "password": "your_password",
    "server": "your_server",
    "demo": True
}
```

---

## 🔮 NEXT STEPS

1. **Добавить API ключи** в config.py
2. **Протестировать TradeLocker** с реальными credentials
3. **Интегрировать с v5 Ultimate** - использовать DataManager для сбора данных
4. **Multi-source training** - обучать модель на данных из разных источников

---

## 📊 DEPENDENCIES

Добавлено в requirements.txt:
```
ccxt>=4.0.0  # Для криптобирж
requests>=2.28.0  # Для TradeLocker
```

---

**Автор:** Golden Breeze AI System
**Commit:** Multi-connector system v1.0

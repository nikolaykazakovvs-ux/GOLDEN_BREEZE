# MT5 Integration for Golden Breeze

Интеграция MetaTrader 5 с системой MCP-серверов Golden Breeze v3.0.

## Компоненты

### 1. MT5 Connector (`mcp_servers/trading/mt5_connector.py`)
Singleton коннектор для управления подключением к MT5:
- `initialize(login, password, server)` — подключение и авторизация
- `shutdown()` — отключение
- `is_connected()` — проверка состояния
- `get_account_info()` — информация об аккаунте

### 2. Market Data Integration (`mcp_servers/trading/market_data.py`)
Получение OHLCV данных из MT5:
- `get_ohlcv(symbol, timeframe, start, end, count)` → DataFrame

Поддерживаемые таймфреймы: M1, M5, M15, M30, H1, H4, D1, W1, MN1

### 3. Trade History Integration (`mcp_servers/trading/trade_history.py`)
Доступ к сделкам и позициям:
- `get_closed_trades(account_id, symbol, start, end)` → List[Dict]
- `get_open_positions(account_id)` → List[Dict]

## Конфигурация

### Вариант 1: Файл конфигурации (рекомендуется)
Заполните `mt5_config.json`:
```json
{
  "login": 123456789,
  "password": "YourPassword",
  "server": "BrokerName-Demo"
}
```

Файл автоматически игнорируется git (`.gitignore`).

### Вариант 2: Прямая передача учётных данных
```python
from mcp_servers.trading.mt5_connector import get_connector

connector = get_connector()
connector.initialize(
    login=123456789,
    password="YourPassword",
    server="BrokerName-Demo"
)
```

## Использование

### Быстрый тест
```powershell
python demo_mt5_integration.py
```

### Получение OHLCV
```python
from mcp_servers.trading import market_data

# Последние 1000 свечей
df = market_data.get_ohlcv("XAUUSD", "M15", count=1000)

# Диапазон дат
df = market_data.get_ohlcv(
    "XAUUSD", 
    "H1", 
    start="2024-01-01",
    end="2024-12-31"
)

# От даты + N свечей
df = market_data.get_ohlcv(
    "EURUSD", 
    "M5", 
    start="2024-11-01",
    count=500
)
```

### История сделок
```python
from mcp_servers.trading import trade_history
from datetime import datetime, timedelta

# Закрытые сделки за последние 30 дней
start = (datetime.now() - timedelta(days=30)).isoformat()
trades = trade_history.get_closed_trades("current", start=start)

# Фильтр по символу
trades = trade_history.get_closed_trades(
    "current",
    symbol="XAUUSD",
    start="2024-11-01",
    end="2024-11-30"
)
```

### Открытые позиции
```python
from mcp_servers.trading import trade_history

positions = trade_history.get_open_positions("current")
for pos in positions:
    print(f"{pos['type']} {pos['volume']} {pos['symbol']} @ {pos['price_open']}")
    print(f"  Profit: {pos['profit']:.2f}")
```

## Интеграция с Golden Breeze

### Обучение моделей на реальных данных
```python
from mcp_servers.trading import market_data
from aimodule.training.train_regime_cluster import train_kmeans

# Получить данные из MT5
df = market_data.get_ohlcv("XAUUSD", "M5", count=10000)

# Обучить модель режимов
train_kmeans(df, n_clusters=4)
```

### Self-Learning из истории MT5
```python
from mcp_servers.trading import trade_history
from aimodule.learning.feedback_store import FeedbackStore

store = FeedbackStore()

# Загрузить историю сделок
trades = trade_history.get_closed_trades("current", start="2024-01-01")

# Преобразовать в feedback
for trade in trades:
    store.add_feedback(
        regime=...,  # определить из context
        direction=trade['type'],
        confidence=...,
        outcome=1 if trade['profit'] > 0 else 0,
        timestamp=trade['time']
    )
```

## Требования
- MetaTrader 5 terminal установлен и запущен
- Пакет `MetaTrader5` Python (уже установлен: v5.0.5388)
- Активное подключение к брокеру (demo или live)

## Безопасность
- `mt5_config.json` добавлен в `.gitignore`
- Не коммитьте учётные данные в репозиторий
- Для production используйте переменные окружения или защищённое хранилище

## Troubleshooting

### "MT5 initialize() failed"
- Убедитесь, что MT5 terminal запущен
- Проверьте, что путь к `terminal64.exe` корректен (если используете custom path)

### "MT5 login failed"
- Проверьте правильность логина/пароля/сервера
- Убедитесь, что аккаунт активен и не заблокирован

### "No data for symbol"
- Проверьте правильность написания символа (например, "XAUUSD" vs "GOLD")
- Убедитесь, что символ добавлен в "Market Watch" в MT5 terminal
- Проверьте доступность исторических данных для символа

## Next Steps
- [ ] Добавить функции для размещения ордеров
- [ ] Интегрировать с `trading.metrics` для расчёта PnL/DD
- [ ] Добавить real-time streaming через MT5 subscriptions
- [ ] Создать адаптер для автоматической торговли через `Connector`

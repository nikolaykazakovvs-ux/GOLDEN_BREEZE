# Trading Metrics Integration

Полная интеграция расчёта торговых метрик для Golden Breeze v3.0.

## Поддерживаемые метрики

### 1. **Date Start / End**
Период тестирования стратегии (начало и конец).

### 2. **ROI (%)** 
Рентабельность инвестиций:
```
ROI = (Net PnL / Initial Balance) × 100
```

### 3. **Net PnL ($)**
Чистая прибыль или убыток за период.

### 4. **Win Ratio (%)**
Процент выигрышных сделок:
```
Win Ratio = (Winning Trades / Total Trades) × 100
```

### 5. **Max Drawdown (%)**
Максимальная просадка от пика к минимуму:
```
Max DD = ((Peak - Trough) / Peak) × 100
```

### 6. **Time in Market (%)**
Процент времени, когда открыта хотя бы одна позиция:
```
Time in Market = (Time in Trades / Total Period) × 100
```

### 7. **Number of Trades**
Общее количество закрытых сделок.

### 8. **Average Trade Duration**
Средняя длительность сделки (от входа до выхода).

### 9. **Timeframe**
Таймфрейм, на котором тестируется стратегия.

## API Functions

### `get_overall_metrics(account_id, start, end, timeframe)`
Получить полный набор метрик.

**Parameters:**
- `account_id` (str): ID аккаунта (или "current")
- `start` (str, optional): Начальная дата ISO (по умолчанию -30 дней)
- `end` (str, optional): Конечная дата ISO (по умолчанию сейчас)
- `timeframe` (str, optional): Таймфрейм стратегии (по умолчанию "M5")

**Returns:** Dict со всеми метриками

**Example:**
```python
from mcp_servers.trading import metrics
from datetime import datetime, timedelta

start = (datetime.now() - timedelta(days=30)).isoformat()

overall = metrics.get_overall_metrics(
    account_id="current",
    start=start,
    timeframe="M5"
)

print(f"ROI: {overall['roi_percent']}%")
print(f"Net PnL: ${overall['net_pnl']}")
print(f"Win Ratio: {overall['win_ratio_percent']}%")
print(f"Max Drawdown: {overall['max_drawdown_percent']}%")
```

### `get_equity_curve(account_id, start, end)`
Получить кривую баланса (equity curve).

**Returns:** List[float] — баланс после каждой сделки

**Example:**
```python
equity = metrics.get_equity_curve("current", start="2024-11-01")

print(f"Start: ${equity[0]:.2f}")
print(f"End: ${equity[-1]:.2f}")
print(f"Peak: ${max(equity):.2f}")
```

### `get_regime_stats(account_id, symbol, regime, start, end)`
Получить статистику по конкретному символу или режиму.

**Example:**
```python
stats = metrics.get_regime_stats(
    account_id="current",
    symbol="XAUUSD",
    regime="Trending",
    start="2024-11-01"
)

print(f"Trades: {stats['trades_count']}")
print(f"Win Ratio: {stats['win_ratio']}%")
print(f"Net PnL: ${stats['net_pnl']}")
```

### `calculate_metrics(trades, initial_balance, timeframe)`
Низкоуровневая функция для расчёта метрик из списка сделок.

**Example:**
```python
from mcp_servers.trading import trade_history, metrics

trades = trade_history.get_closed_trades("current", start="2024-11-01")
result = metrics.calculate_metrics(trades, initial_balance=10000, timeframe="M15")

print(result)
```

## Usage Examples

### Пример 1: Базовый отчёт

```python
from mcp_servers.trading import metrics
from datetime import datetime, timedelta

# Метрики за последние 30 дней
start = (datetime.now() - timedelta(days=30)).isoformat()

overall = metrics.get_overall_metrics("current", start=start, timeframe="M5")

print(f"""
Trading Metrics Report
======================
Period:       {overall['date_start']} → {overall['date_end']}
Timeframe:    {overall['timeframe']}

Performance:
  Net PnL:    ${overall['net_pnl']:.2f}
  ROI:        {overall['roi_percent']:.2f}%

Statistics:
  Trades:     {overall['number_of_trades']}
  Win Ratio:  {overall['win_ratio_percent']:.2f}%
  Avg Duration: {overall['average_trade_duration']}

Risk:
  Max DD:     {overall['max_drawdown_percent']:.2f}%
  Time in Market: {overall['time_in_market_percent']:.2f}%
""")
```

### Пример 2: Сравнение периодов

```python
from datetime import datetime, timedelta
from mcp_servers.trading import metrics

# Последние 30 дней
start_30 = (datetime.now() - timedelta(days=30)).isoformat()
metrics_30d = metrics.get_overall_metrics("current", start=start_30)

# Последние 7 дней
start_7 = (datetime.now() - timedelta(days=7)).isoformat()
metrics_7d = metrics.get_overall_metrics("current", start=start_7)

print(f"""
Comparison: 30d vs 7d
=====================
                   30 days      7 days
Trades:           {metrics_30d['number_of_trades']:>6}      {metrics_7d['number_of_trades']:>6}
Net PnL:         ${metrics_30d['net_pnl']:>7.2f}    ${metrics_7d['net_pnl']:>7.2f}
Win Ratio:        {metrics_30d['win_ratio_percent']:>5.1f}%      {metrics_7d['win_ratio_percent']:>5.1f}%
ROI:              {metrics_30d['roi_percent']:>5.1f}%      {metrics_7d['roi_percent']:>5.1f}%
""")
```

### Пример 3: Сохранение в JSON

```python
from mcp_servers.trading import metrics
from datetime import datetime, timedelta
import json

start = (datetime.now() - timedelta(days=30)).isoformat()
overall = metrics.get_overall_metrics("current", start=start)

# Сохранить отчёт
with open('metrics_report.json', 'w', encoding='utf-8') as f:
    json.dump(overall, f, indent=2, ensure_ascii=False)

print("✓ Metrics saved to metrics_report.json")
```

### Пример 4: Построение графиков (требует matplotlib)

```python
import matplotlib.pyplot as plt
from mcp_servers.trading import metrics
from datetime import datetime, timedelta

start = (datetime.now() - timedelta(days=30)).isoformat()

# Получить equity curve
equity = metrics.get_equity_curve("current", start=start)

# Построить график
plt.figure(figsize=(12, 6))
plt.plot(equity, linewidth=2, color='#2E86DE')
plt.fill_between(range(len(equity)), equity, alpha=0.3)
plt.title('Equity Curve', fontsize=16, fontweight='bold')
plt.xlabel('Trade Number')
plt.ylabel('Balance ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=150)
plt.show()
```

## Demo Scripts

### 1. Basic Metrics Demo
```powershell
python demo_trading_metrics.py
```

Показывает полный набор метрик и сохраняет отчёт в JSON.

### 2. Visualization (требует matplotlib)
```powershell
# Установить matplotlib
pip install matplotlib

# Запустить визуализацию
python visualize_metrics.py
```

Создаёт графики:
- `equity_curve.png` — кривая баланса с отметками сделок
- `metrics_summary.png` — сводная панель метрик

## Integration с Golden Breeze

### Использование метрик для self-learning

```python
from mcp_servers.trading import metrics, trade_history
from aimodule.learning.feedback_store import FeedbackStore

# Получить метрики по символу
stats = metrics.get_regime_stats("current", symbol="XAUUSD", regime="Trending")

# Если win ratio низкий, скорректировать пороги
if stats['win_ratio'] < 50:
    print("⚠️  Win ratio below 50%, adjusting thresholds...")
    # Обновить OnlineUpdater
```

### Мониторинг производительности

```python
from mcp_servers.trading import metrics
import time

while True:
    # Обновлять метрики каждый час
    overall = metrics.get_overall_metrics("current", timeframe="M5")
    
    # Алерт при высоком drawdown
    if overall['max_drawdown_percent'] > 20:
        print(f"⚠️  High drawdown: {overall['max_drawdown_percent']:.2f}%")
        # Отправить уведомление / остановить торговлю
    
    time.sleep(3600)
```

## Output Format

### get_overall_metrics() возвращает:
```python
{
    "date_start": "2025-11-25T06:08:22+02:00",
    "date_end": "2025-12-01T05:15:00+02:00",
    "roi_percent": 99.95,
    "net_pnl": 10399.22,
    "win_ratio_percent": 40.00,
    "max_drawdown_percent": 0.07,
    "time_in_market_percent": 0.00,
    "number_of_trades": 10,
    "average_trade_duration": "2h 15m",
    "timeframe": "M5",
    "equity_curve": [10000, 10100, 10050, ...],
    "account_info": {
        "login": 99332338,
        "server": "MetaQuotes-Demo",
        "current_balance": 10404.03,
        "current_equity": 10404.03,
        "currency": "USD"
    }
}
```

## Performance Indicators

### ✓ Good Performance
- ROI > 10% (annual)
- Win Ratio > 50%
- Max Drawdown < 20%
- Consistent equity growth

### ⚠ Warning Signs
- Win Ratio < 40%
- Max Drawdown > 30%
- Negative ROI
- Erratic equity curve

### 📊 Benchmarks (для M5 стратегии)
- Excellent: ROI > 50%, Win Ratio > 60%, Max DD < 15%
- Good: ROI > 20%, Win Ratio > 50%, Max DD < 25%
- Acceptable: ROI > 10%, Win Ratio > 45%, Max DD < 35%
- Poor: ROI < 5%, Win Ratio < 40%, Max DD > 40%

## Requirements

- MetaTrader 5 с историей сделок
- Минимум 10 закрытых сделок для статистически значимых результатов
- (Optional) matplotlib для визуализации: `pip install matplotlib`

## Next Steps

1. Добавить real-time мониторинг метрик
2. Интегрировать с Telegram/Email уведомлениями
3. Добавить A/B тестирование стратегий
4. Создать web dashboard для метрик

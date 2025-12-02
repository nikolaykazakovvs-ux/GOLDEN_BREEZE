# MCP Servers

This folder contains MCP server implementations described in `MCP_SERVERS_GOLDEN_BREEZE.md`.

They provide standardized APIs for agents to interact with code, data, market, and operations.

## Structure

- **core**: fs, git, shell, python_runtime
- **trading**: market_data ✓ MT5, trade_history ✓ MT5, news, metrics
- **ops**: logs, config, cicd

✓ = Real integration implemented

## MT5 Integration (LIVE)

Market data, trade history, and **trading metrics** are now connected to MetaTrader 5!

```python
from mcp_servers.trading import market_data, trade_history, metrics

# Get real OHLCV data from MT5
df = market_data.get_ohlcv("XAUUSD", "M15", count=1000)

# Get closed trades and open positions
trades = trade_history.get_closed_trades("current", start="2024-11-01")
positions = trade_history.get_open_positions("current")

# Get comprehensive trading metrics
overall = metrics.get_overall_metrics("current", start="2024-11-01", timeframe="M5")
print(f"ROI: {overall['roi_percent']}%")
print(f"Win Ratio: {overall['win_ratio_percent']}%")
print(f"Max Drawdown: {overall['max_drawdown_percent']}%")
```

See `docs/MT5_INTEGRATION.md` and `docs/METRICS_INTEGRATION.md` for full documentation.

## Quick usage examples

### Core Servers
```python
from mcp_servers.core import fs, git, shell, python_runtime

print(fs.list("aimodule"))
print(git.git_status())
print(shell.run("python -m pytest -q"))
print(python_runtime.python_exec("result = 1+1"))
```

### Trading Servers (MT5 Live)
```python
from mcp_servers.trading import market_data, trade_history

# Real market data
df = market_data.get_ohlcv("XAUUSD", "M15", count=100)
df_range = market_data.get_ohlcv("EURUSD", "H1", start="2024-11-01", end="2024-11-30")

# Trade history
closed = trade_history.get_closed_trades("current", symbol="XAUUSD", start="2024-11-01")
positions = trade_history.get_open_positions("current")
```

### News & Metrics (stubs)
```python
from mcp_servers.trading import news, metrics

headlines = news.get_news("XAUUSD", limit=5)  # stub
overall = metrics.get_overall_metrics("acc-001")  # stub
```

## Test MT5 Integration

```powershell
python demo_mt5_integration.py
```

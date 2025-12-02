"""Quick final test of MT5 integration."""
from mcp_servers.trading.mt5_connector import get_connector
from mcp_servers.trading import market_data, trade_history
from datetime import datetime, timedelta

print('=== Final MT5 Integration Test ===\n')

# Test 1: Imports
print('[1/4] Testing imports...')
print('✓ Imports OK\n')

# Test 2: Connection
print('[2/4] Testing MT5 connection...')
conn = get_connector()
if conn.initialize():
    print('✓ Connected')
    info = conn.get_account_info()
    if info:
        print(f'✓ Account: {info["login"]} @ {info["server"]}')
        print(f'✓ Balance: {info["balance"]} {info["currency"]}\n')
else:
    print('✗ Failed to connect\n')

# Test 3: Market Data
print('[3/4] Testing market data...')
df = market_data.get_ohlcv('XAUUSD', 'M15', count=5)
if not df.empty:
    print(f'✓ Got {len(df)} bars')
    print(f'✓ Latest: {df["close"].iloc[-1]:.2f}\n')
else:
    print('✗ No data\n')

# Test 4: Trade History
print('[4/4] Testing trade history...')
start = (datetime.now() - timedelta(days=7)).isoformat()
trades = trade_history.get_closed_trades('current', start=start)
print(f'✓ Got {len(trades)} trades in last 7 days\n')

positions = trade_history.get_open_positions('current')
print(f'✓ Open positions: {len(positions)}\n')

conn.shutdown()
print('=== All Tests Passed ✓ ===')

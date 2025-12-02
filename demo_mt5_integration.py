"""Demo script: MT5 integration test for Golden Breeze MCP servers.

Demonstrates:
- Connection to MT5
- Fetching OHLCV data
- Getting account info
- Retrieving open positions and trade history
"""
from mcp_servers.trading.mt5_connector import get_connector
from mcp_servers.trading import market_data, trade_history

def main():
    print("=" * 60)
    print("Golden Breeze MT5 Integration Test")
    print("=" * 60)
    
    # 1. Подключение к MT5
    print("\n[1] Connecting to MT5...")
    connector = get_connector()
    
    # Можно передать учётные данные напрямую:
    # success = connector.initialize(login=123456, password="pass", server="Broker-Demo")
    
    # Или загрузить из mt5_config.json:
    success = connector.initialize()
    
    if not success:
        print("❌ Failed to connect to MT5")
        print("Please:")
        print("  1. Ensure MT5 terminal is installed and running")
        print("  2. Fill mt5_config.json with your credentials, or")
        print("  3. Pass credentials directly to connector.initialize()")
        return
    
    print("✓ MT5 connected successfully")
    
    # 2. Информация об аккаунте
    print("\n[2] Account Info:")
    account_info = connector.get_account_info()
    if account_info:
        print(f"  Login:    {account_info['login']}")
        print(f"  Server:   {account_info['server']}")
        print(f"  Balance:  {account_info['balance']} {account_info['currency']}")
        print(f"  Equity:   {account_info['equity']} {account_info['currency']}")
        print(f"  Profit:   {account_info['profit']} {account_info['currency']}")
        print(f"  Leverage: 1:{account_info['leverage']}")
    
    # 3. Получение OHLCV данных
    print("\n[3] Fetching OHLCV data (XAUUSD, M15, last 100 bars)...")
    df = market_data.get_ohlcv("XAUUSD", "M15", count=100)
    
    if not df.empty:
        print(f"✓ Retrieved {len(df)} bars")
        print(f"  Date range: {df['time'].min()} → {df['time'].max()}")
        print(f"  Latest close: {df['close'].iloc[-1]:.2f}")
        print("\nFirst 5 rows:")
        print(df.head())
    else:
        print("❌ No OHLCV data received")
    
    # 4. Открытые позиции
    print("\n[4] Open Positions:")
    positions = trade_history.get_open_positions("current")
    
    if positions:
        print(f"✓ Found {len(positions)} open position(s)")
        for pos in positions:
            print(f"  #{pos['ticket']}: {pos['type']} {pos['volume']} {pos['symbol']} @ {pos['price_open']:.2f}")
            print(f"    Current: {pos['price_current']:.2f}, Profit: {pos['profit']:.2f}")
    else:
        print("  No open positions")
    
    # 5. История сделок (последние 7 дней)
    print("\n[5] Closed Trades (last 7 days):")
    from datetime import datetime, timedelta
    start = (datetime.now() - timedelta(days=7)).isoformat()
    trades = trade_history.get_closed_trades("current", start=start)
    
    if trades:
        print(f"✓ Found {len(trades)} closed trade(s)")
        for trade in trades[:5]:  # Показываем первые 5
            print(f"  #{trade['ticket']}: {trade['type']} {trade['volume']} {trade['symbol']} @ {trade['price']:.2f}")
            print(f"    Profit: {trade['profit']:.2f}, Time: {trade['time']}")
    else:
        print("  No closed trades in the period")
    
    # 6. Отключение
    print("\n[6] Disconnecting from MT5...")
    connector.shutdown()
    print("✓ Disconnected")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

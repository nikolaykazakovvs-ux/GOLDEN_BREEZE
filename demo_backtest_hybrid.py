# demo_backtest_hybrid.py
"""
Демонстрация backtesting гибридной стратегии
"""

from strategy import StrategyConfig, HybridStrategy
from strategy.backtest_engine import BacktestEngine
from mcp_servers.trading import market_data, MT5Connector
import pandas as pd
import numpy as np


def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление индикаторов к данным
    
    Требуются: SMA, ATR, RSI
    """
    data = df.copy()
    
    # SMA Fast (20)
    data["sma_fast"] = data["close"].rolling(window=20).mean()
    
    # SMA Slow (50)
    data["sma_slow"] = data["close"].rolling(window=50).mean()
    
    # ATR (14)
    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(window=14).mean()
    
    # RSI (14)
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))
    
    # Заполнение NaN
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data


def demo_backtest_from_mt5():
    """
    Backtest на данных из MT5
    """
    print("\n" + "="*60)
    print("Golden Breeze Hybrid Strategy - Backtest Demo")
    print("="*60)
    
    # Подключение к MT5
    print("\n[1] Connecting to MT5...")
    connector = MT5Connector()
    if not connector.initialize():
        print("❌ Failed to connect to MT5")
        return
    
    print("✅ Connected to MT5")
    account_info = connector.get_account_info()
    print(f"   Account: {account_info.get('login', 'N/A')}")
    print(f"   Server: {account_info.get('server', 'N/A')}")
    
    # Получение M5 данных
    print("\n[2] Loading M5 data...")
    m5_data = market_data.get_ohlcv("XAUUSD", "M5", count=5000)
    
    if m5_data is None or len(m5_data) < 100:
        print("❌ Failed to load M5 data")
        connector.shutdown()
        return
    
    print(f"✅ Loaded {len(m5_data)} M5 bars")
    print(f"   Period: {m5_data.index[0]} to {m5_data.index[-1]}")
    
    # Добавление индикаторов
    print("\n[3] Calculating indicators...")
    m5_data = prepare_data_with_indicators(m5_data)
    print("✅ Indicators added: SMA, ATR, RSI")
    
    # Получение M1 данных (опционально)
    print("\n[4] Loading M1 data for intrabar simulation...")
    m1_data = market_data.get_ohlcv("XAUUSD", "M1", count=10000)
    
    if m1_data is not None and len(m1_data) > 0:
        print(f"✅ Loaded {len(m1_data)} M1 bars")
    else:
        print("⚠️  M1 data not available, using simple simulation")
        m1_data = None
    
    # Конфигурация стратегии
    print("\n[5] Configuring strategy...")
    config = StrategyConfig(
        symbol="XAUUSD",
        base_timeframe="M5",
        risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        max_total_dd_pct=10.0,
        max_positions=2,
        min_direction_confidence=0.65,
        ai_api_url="http://127.0.0.1:5005",
        
        # Trend settings
        trend_partial_tp_pct=50.0,
        trend_trailing_atr_mult=2.0,
        trend_min_profit_for_trail=0.5,
        
        # Range settings
        range_tp_fixed_points=100.0,
        range_max_atr_threshold=150.0,
        
        # Volatile settings
        volatile_allow_trades=False,
        volatile_risk_reduction=0.5,
        
        # Backtesting
        use_tick_data=False,  # Используем M1 или простую симуляцию
        initial_balance=10000.0
    )
    
    print("✅ Strategy configured")
    print(f"   Initial Balance: ${config.initial_balance:,.2f}")
    print(f"   Risk per Trade: {config.risk_per_trade_pct}%")
    print(f"   Max Daily Loss: {config.max_daily_loss_pct}%")
    print(f"   Max Drawdown: {config.max_total_dd_pct}%")
    
    # Создание стратегии
    print("\n[6] Initializing strategy...")
    strategy = HybridStrategy(config, initial_balance=config.initial_balance)
    
    # Проверка AI сервера
    if strategy.ai_client.health_check():
        print("✅ AI Core connected")
    else:
        print("⚠️  AI Core not available - strategy will use default logic")
    
    # Backtesting engine
    print("\n[7] Creating backtest engine...")
    backtest = BacktestEngine(strategy, config)
    backtest.load_m5_data(m5_data)
    
    if m1_data is not None:
        backtest.load_m1_data(m1_data)
    
    print("✅ Backtest engine ready")
    
    # Запуск backtesting
    print("\n[8] Running backtest...")
    print("   (This may take a few minutes...)\n")
    
    # Используем последние 1000 баров для быстрого теста
    start_date = m5_data.index[-1000].strftime("%Y-%m-%d")
    end_date = m5_data.index[-1].strftime("%Y-%m-%d")
    
    backtest.run(start_date=start_date, end_date=end_date)
    
    # Экспорт результатов
    print("\n[9] Exporting results...")
    backtest.export_results("backtest_hybrid_results.csv")
    
    # Equity curve
    equity_df = backtest.get_equity_curve()
    equity_df.to_csv("backtest_equity_curve.csv", index=False)
    
    print("✅ Results exported:")
    print("   - backtest_hybrid_results.csv (trades)")
    print("   - backtest_equity_curve.csv (equity)")
    
    # Отключение MT5
    connector.shutdown()
    print("\n✅ Backtest complete!")


def demo_backtest_from_csv():
    """
    Backtest на данных из CSV файла
    """
    print("\n" + "="*60)
    print("Golden Breeze Hybrid Strategy - Backtest from CSV")
    print("="*60)
    
    # Загрузка данных
    print("\n[1] Loading data from CSV...")
    try:
        m5_data = pd.read_csv("xauusd_m5.csv", index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(m5_data)} M5 bars from CSV")
    except FileNotFoundError:
        print("❌ File xauusd_m5.csv not found")
        print("   Please prepare your CSV file with columns: timestamp, open, high, low, close, volume")
        return
    
    # Добавление индикаторов
    print("\n[2] Calculating indicators...")
    m5_data = prepare_data_with_indicators(m5_data)
    
    # Конфигурация
    config = StrategyConfig(
        symbol="XAUUSD",
        risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        initial_balance=10000.0
    )
    
    # Стратегия
    strategy = HybridStrategy(config, initial_balance=10000.0)
    
    # Backtest
    backtest = BacktestEngine(strategy, config)
    backtest.load_m5_data(m5_data)
    
    # Запуск
    backtest.run()
    
    # Экспорт
    backtest.export_results("backtest_csv_results.csv")
    
    print("\n✅ Backtest from CSV complete!")


if __name__ == "__main__":
    import sys
    
    print("\n🚀 Golden Breeze Hybrid Strategy - Backtest Demo\n")
    print("Select data source:")
    print("  1. MT5 (live connection)")
    print("  2. CSV file (xauusd_m5.csv)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_backtest_from_mt5()
    elif choice == "2":
        demo_backtest_from_csv()
    else:
        print("Invalid choice")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMC Model Backtest Script

Tests the new Direction LSTM model with SMC features (direction_lstm_smc_v1.pt)
using the Golden Breeze Hybrid Strategy backtest engine.

Usage:
    python tools/run_smc_backtest.py

Author: Golden Breeze Team
Version: 1.0 (SMC Integration)
"""

import sys
from pathlib import Path
import pandas as pd

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from strategy.ai_client import AIClient
from strategy.hybrid_strategy import HybridStrategy
from strategy.backtest_engine import BacktestEngine
from strategy.config import StrategyConfig


def main():
    """Run SMC model backtest."""
    
    print("=" * 70)
    print("Golden Breeze - SMC Model Backtest")
    print("=" * 70)
    print(f"Model: models/direction_lstm_smc_v1.pt")
    print(f"Dataset: Last 3000 candles from data/raw/XAUUSD/M5.csv")
    print(f"AI Min Confidence: 0.60 (Testing threshold)")
    print("=" * 70 + "\n")
    
    # STEP 1: Check if AI server is running
    print("1. Checking AI server status...")
    ai_client = AIClient(api_url="http://127.0.0.1:5005")
    
    if not ai_client.health_check():
        print("❌ AI server is not running!")
        print("\nPlease start the AI server first:")
        print("  Option 1 (PowerShell): .\\run_server.ps1")
        print("  Option 2 (Manual): python -m uvicorn aimodule.server.local_ai_gateway:app --host 127.0.0.1 --port 5005")
        print("\nNote: Make sure to update DIRECTION_LSTM_MODEL_PATH in aimodule/config.py")
        print("      to point to 'models/direction_lstm_smc_v1.pt' before starting server")
        return 1
    
    print("✅ AI server is running")
    
    # STEP 2: Configure strategy with lowered confidence threshold
    print("\n2. Configuring strategy parameters...")
    config = StrategyConfig(
        symbol="XAUUSD",
        primary_tf="M5",
        initial_balance=10000.0,
        min_direction_confidence=0.60,  # Lower threshold to test signal frequency
        ai_api_url="http://127.0.0.1:5005"
    )
    print(f"✅ min_direction_confidence set to {config.min_direction_confidence}")
    
    # STEP 3: Initialize HybridStrategy
    print("\n3. Initializing HybridStrategy...")
    try:
        strategy = HybridStrategy(config=config, initial_balance=config.initial_balance)
        print("✅ HybridStrategy initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize HybridStrategy: {e}")
        return 1
    
    # STEP 4: Load M5 data
    print("\n4. Loading XAUUSD M5 data...")
    data_path = Path("data/raw/XAUUSD/M5.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1
    
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Take last 3000 candles
        df = df.tail(3000)
        
        print(f"✅ Loaded {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # STEP 5: Initialize BacktestEngine
    print("\n5. Initializing BacktestEngine...")
    try:
        backtest_engine = BacktestEngine(strategy=strategy, config=config)
        backtest_engine.load_m5_data(df)
        
        print("✅ BacktestEngine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize BacktestEngine: {e}")
        return 1
    
    # STEP 6: Run backtest
    print("\n6. Running backtest...")
    print("=" * 70)
    
    try:
        backtest_engine.run()
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 7: Extract and display results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS - TRADE LOG")
    print("=" * 70)
    
    trade_history = strategy.risk_manager.trade_history
    
    if not trade_history:
        print("⚠ No trades executed during backtest period")
        return 0
    
    # Display trade table
    print(f"\n{'ID':<5} {'Time':<20} {'Type':<6} {'Entry':<10} {'Exit':<10} {'PnL':<12} {'Regime':<12}")
    print("-" * 85)
    
    total_pnl = 0.0
    winning_trades = 0
    
    for trade in trade_history:
        trade_id = trade.id
        entry_time = str(trade.entry_time)[:19] if trade.entry_time else "N/A"
        direction = trade.direction[:4].upper()  # BUY/SELL
        entry_price = f"{trade.entry_price:.2f}"
        exit_price = f"{trade.exit_price:.2f}" if trade.exit_price else "OPEN"
        pnl = trade.pnl if trade.pnl is not None else 0.0
        regime = trade.regime or "N/A"
        
        # Color code PnL
        pnl_str = f"${pnl:+.2f}"
        
        print(f"{trade_id:<5} {entry_time:<20} {direction:<6} {entry_price:<10} {exit_price:<10} {pnl_str:<12} {regime:<12}")
        
        total_pnl += pnl
        if pnl > 0:
            winning_trades += 1
    
    # Summary statistics
    print("-" * 85)
    total_trades = len(trade_history)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total Trades:      {total_trades}")
    print(f"Winning Trades:    {winning_trades}")
    print(f"Losing Trades:     {total_trades - winning_trades}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"Total Net Profit:  ${total_pnl:+,.2f}")
    print(f"Final Balance:     ${strategy.risk_manager.current_balance:,.2f}")
    print(f"ROI:               {(total_pnl / config.initial_balance * 100):+.2f}%")
    print("=" * 70)
    
    # Additional metrics from risk manager
    print(f"\n📊 Risk Manager Metrics:")
    print(f"   Initial Balance:  ${config.initial_balance:,.2f}")
    print(f"   Final Balance:    ${strategy.risk_manager.current_balance:,.2f}")
    print(f"   Max Drawdown:     {strategy.risk_manager.max_drawdown:.2f}%")
    
    print("\n✅ Backtest completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

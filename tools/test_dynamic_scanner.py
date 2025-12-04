"""
Test script for Dynamic Timeframe Scanner
Tests the new Smart TF Scanner and High Confidence Override features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
from strategy.hybrid_strategy import HybridStrategy
from strategy.config import StrategyConfig
from strategy.backtest_engine import BacktestEngine


def test_dynamic_scanner():
    """Test Dynamic Timeframe Scanner"""
    
    print("="*70)
    print("🧪 TEST: Dynamic Timeframe Scanner")
    print("="*70)
    
    # Step 1: Initialize Strategy with TF Selector enabled
    config = StrategyConfig(
        symbol="XAUUSD",
        primary_tf="M5",
        tf_selector_enable=True,
        tf_selector_min_confidence=0.65,
        tf_selector_high_confidence=0.85,
        ai_api_url="http://127.0.0.1:5005"
    )
    
    strategy = HybridStrategy(config, initial_balance=10000.0)
    print("✅ Strategy initialized with TF Selector enabled")
    
    # Step 2: Test Smart TF Scanner
    print("\n" + "="*70)
    print("🔍 Testing Smart Timeframe Scanner")
    print("="*70)
    
    if strategy.tf_selector:
        try:
            best_tf = strategy.tf_selector.scan_best_timeframe(
                symbol="XAUUSD",
                ai_client=strategy.ai_client
            )
            print(f"\n✅ Scanner selected: {best_tf}")
        except Exception as e:
            print(f"\n❌ Scanner failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ TF Selector not enabled!")
    
    # Step 3: Load test data and run short backtest
    print("\n" + "="*70)
    print("📊 Running Short Backtest on Selected TF")
    print("="*70)
    
    try:
        # Try to load CSV data (assuming it exists)
        csv_path = project_root / "data" / "prepared" / "XAUUSD_M5.csv"
        
        if csv_path.exists():
            engine = BacktestEngine(strategy, config)
            
            # Load data with proper date parsing (FIX applied)
            df = engine.load_csv_data(str(csv_path), timeframe="M5")
            
            # Run backtest on last 500 candles
            print(f"\n🚀 Running backtest on {len(df)} candles...")
            engine.run()
            
            # Show results
            print("\n" + "="*70)
            print("📈 BACKTEST RESULTS")
            print("="*70)
            
            metrics = engine.get_metrics()
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Total PnL: ${metrics.get('total_pnl', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            if metrics.get('total_trades', 0) > 0:
                print("\n✅ SUCCESS: Backtest generated trades!")
            else:
                print("\n⚠️  WARNING: 0 trades generated (check AI server)")
        else:
            print(f"❌ Data file not found: {csv_path}")
            print("   Please run data preparation first")
    
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 Test Complete")
    print("="*70)


def test_high_confidence_override():
    """Test High Confidence Override logic"""
    
    print("\n\n" + "="*70)
    print("🧪 TEST: High Confidence Override")
    print("="*70)
    
    # Create mock data
    mock_data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': [2000.0 + i * 0.1 for i in range(100)],
        'high': [2000.5 + i * 0.1 for i in range(100)],
        'low': [1999.5 + i * 0.1 for i in range(100)],
        'close': [2000.2 + i * 0.1 for i in range(100)],
        'volume': [100] * 100
    })
    mock_data.set_index('time', inplace=True)
    
    config = StrategyConfig(
        symbol="XAUUSD",
        primary_tf="M5",
        ai_api_url="http://127.0.0.1:5005"
    )
    
    strategy = HybridStrategy(config, initial_balance=10000.0)
    
    # Test Case 1: High confidence (should override regime)
    print("\n📌 Test Case 1: High Confidence = 0.95 (should override)")
    mock_signal = {
        "regime": "range",  # Normally would avoid
        "direction": "long",
        "direction_confidence": 0.95  # High confidence!
    }
    
    signal = strategy._generate_trading_signal(mock_data, mock_signal)
    
    if signal:
        print(f"✅ Signal generated: {signal['type']}")
        print(f"   Reason: {signal['reason']}")
        print(f"   Confidence: {signal['confidence']:.2f}")
    else:
        print("❌ No signal (unexpected!)")
    
    # Test Case 2: Low confidence (should NOT override)
    print("\n📌 Test Case 2: Low Confidence = 0.70 (should follow regime)")
    mock_signal = {
        "regime": "unknown",
        "direction": "long",
        "direction_confidence": 0.70  # Normal confidence
    }
    
    signal = strategy._generate_trading_signal(mock_data, mock_signal)
    
    if signal:
        print(f"⚠️  Signal generated (may be from regime strategy)")
        print(f"   Reason: {signal.get('reason', 'N/A')}")
    else:
        print("✅ No signal (correct - low confidence + bad regime)")
    
    print("\n" + "="*70)
    print("🏁 Override Test Complete")
    print("="*70)


if __name__ == "__main__":
    # Run both tests
    test_dynamic_scanner()
    test_high_confidence_override()
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS COMPLETE")
    print("="*70)

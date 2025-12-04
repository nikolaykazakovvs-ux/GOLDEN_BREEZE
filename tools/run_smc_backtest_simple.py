#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple SMC Model Backtest - Lightweight version

Tests the SMC model on a small dataset without multitimeframe overhead.

Usage:
    python tools/run_smc_backtest_simple.py

Author: Golden Breeze Team
Version: 1.0
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from strategy.ai_client import AIClient
from strategy.config import StrategyConfig


def main():
    """Run simplified SMC backtest."""
    
    print("=" * 70)
    print("Golden Breeze - SMC Model Simple Backtest")
    print("=" * 70)
    print(f"Model: models/direction_lstm_smc_v1.pt")
    print(f"Dataset: Last 100 candles (lightweight test)")
    print("=" * 70 + "\n")
    
    # STEP 1: Check AI server
    print("1. Checking AI server status...")
    ai_client = AIClient(api_url="http://127.0.0.1:5005")
    
    if not ai_client.health_check():
        print("❌ AI server is not running!")
        return 1
    
    print("✅ AI server is running")
    
    # STEP 2: Load data
    print("\n2. Loading XAUUSD M5 data...")
    data_path = Path("data/raw/XAUUSD/M5.csv")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1
    
    try:
        df = pd.read_csv(data_path)
        
        # Take last 100 candles for quick test
        df = df.tail(100)
        
        print(f"✅ Loaded {len(df)} candles")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    # STEP 3: Test AI predictions
    print("\n3. Testing AI predictions...")
    print("-" * 70)
    
    successful_predictions = 0
    failed_predictions = 0
    predictions_summary = {
        "long": 0,
        "short": 0,
        "flat": 0
    }
    
    # Test on every 10th candle to reduce load
    test_indices = range(50, len(df), 10)
    
    for i in test_indices:
        # Prepare candles for prediction (last 50)
        window_start = max(0, i - 50)
        candles = []
        
        for idx in range(window_start, i):
            row = df.iloc[idx]
            candles.append({
                "timestamp": str(row.get('time', f"2025-11-{idx:02d} 00:00:00")),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row.get('volume', 0))
            })
        
        # Get prediction
        try:
            signal = ai_client.predict(
                symbol="XAUUSD",
                timeframe="M5",
                candles=candles
            )
            
            if signal:
                successful_predictions += 1
                direction = signal.get('direction', 'flat')
                confidence = signal.get('confidence', 0.0)  # FIX: использовать 'confidence' вместо 'direction_confidence'
                regime = signal.get('regime', 'unknown')
                
                predictions_summary[direction] = predictions_summary.get(direction, 0) + 1
                
                print(f"  Candle {i:3d}: {direction.upper():<6} (conf: {confidence:.2f}, regime: {regime})")
            else:
                failed_predictions += 1
                print(f"  Candle {i:3d}: ❌ No signal")
            
            # Small delay to avoid overwhelming server
            time.sleep(0.1)
            
        except Exception as e:
            failed_predictions += 1
            print(f"  Candle {i:3d}: ❌ Error: {e}")
    
    # STEP 4: Summary
    print("\n" + "=" * 70)
    print("SIMPLE BACKTEST RESULTS")
    print("=" * 70)
    
    total_tests = len(test_indices)
    print(f"\nTotal Tests:         {total_tests}")
    print(f"Successful:          {successful_predictions} ({successful_predictions/total_tests*100:.1f}%)")
    print(f"Failed:              {failed_predictions} ({failed_predictions/total_tests*100:.1f}%)")
    
    print(f"\nPrediction Distribution:")
    print(f"  LONG:              {predictions_summary.get('long', 0)}")
    print(f"  SHORT:             {predictions_summary.get('short', 0)}")
    print(f"  FLAT:              {predictions_summary.get('flat', 0)}")
    
    if successful_predictions > 0:
        long_pct = predictions_summary.get('long', 0) / successful_predictions * 100
        short_pct = predictions_summary.get('short', 0) / successful_predictions * 100
        flat_pct = predictions_summary.get('flat', 0) / successful_predictions * 100
        
        print(f"\nPercentages:")
        print(f"  LONG:              {long_pct:.1f}%")
        print(f"  SHORT:             {short_pct:.1f}%")
        print(f"  FLAT:              {flat_pct:.1f}%")
    
    print("\n" + "=" * 70)
    
    if successful_predictions > 0:
        print("✅ SMC Model is working and generating predictions!")
        return 0
    else:
        print("❌ No successful predictions - check AI server and model")
        return 1


if __name__ == "__main__":
    sys.exit(main())

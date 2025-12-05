"""
Export strategy tester history from MT5.
MT5 Strategy Tester has full history (10+ years) for all symbols.
"""

import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
from datetime import datetime

def export_tester_history():
    """Export full history available in Strategy Tester."""
    
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return
    
    print("=" * 60)
    print("Exporting from MT5 Strategy Tester History")
    print("=" * 60)
    
    symbol = "XAUUSD"
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        mt5.shutdown()
        return
    
    print(f"\n📊 Symbol: {symbol}")
    
    # Try to get maximum available data using copy_rates_from_pos
    # Start from position 0 (most recent) and go back as far as possible
    
    timeframes = [
        ("M5", mt5.TIMEFRAME_M5, 500000),  # Request 500k bars
        ("H1", mt5.TIMEFRAME_H1, 100000),
    ]
    
    output_dir = Path("data/raw/XAUUSD")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for tf_name, tf_constant, max_bars in timeframes:
        print(f"\n📥 {tf_name}:")
        print(f"   Requesting up to {max_bars:,} bars...")
        
        # Use copy_rates_from_pos to get as much history as available
        rates = mt5.copy_rates_from_pos(symbol, tf_constant, 0, max_bars)
        
        if rates is None or len(rates) == 0:
            print(f"   ❌ No data")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Save
        output_file = output_dir / f"{tf_name}_full.csv"
        df.to_csv(output_file, index=False)
        
        print(f"   ✅ Got {len(df):,} bars")
        print(f"   Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        print(f"   💾 Saved: {output_file}")
        
        # Calculate time span
        days = (df['time'].iloc[-1] - df['time'].iloc[0]).days
        years = days / 365.25
        print(f"   📅 Span: {days} days ({years:.1f} years)")
    
    mt5.shutdown()
    print("\n✅ Export complete!")
    print("\n💡 Note: If you need MORE data, enable 'Max bars in chart'")
    print("   in MT5: Tools -> Options -> Charts -> Max bars in chart: 999999999")


if __name__ == "__main__":
    export_tester_history()

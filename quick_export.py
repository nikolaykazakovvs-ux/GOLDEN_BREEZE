"""Quick export M1 and M5 with direct MT5 API"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    exit(1)

print("✅ MT5 connected")

# Dates
end = datetime.now()
start = end - timedelta(days=180)  # 6 months

print(f"Period: {start.date()} to {end.date()}")

# Export M5
print("\n📊 Exporting M5...")
rates_m5 = mt5.copy_rates_range('XAUUSD', mt5.TIMEFRAME_M5, start, end)
if rates_m5 is not None and len(rates_m5) > 0:
    df_m5 = pd.DataFrame(rates_m5)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s', utc=True)
    
    output_path = Path('data/raw/XAUUSD/M5.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_m5.to_csv(output_path, index=True)
    
    print(f"✅ M5: {len(df_m5)} bars saved")
    print(f"   Period: {df_m5['time'].iloc[0]} to {df_m5['time'].iloc[-1]}")
else:
    print("❌ M5: No data")

# Export M1
print("\n📊 Exporting M1...")
rates_m1 = mt5.copy_rates_range('XAUUSD', mt5.TIMEFRAME_M1, start, end)
if rates_m1 is not None and len(rates_m1) > 0:
    df_m1 = pd.DataFrame(rates_m1)
    df_m1['time'] = pd.to_datetime(df_m1['time'], unit='s', utc=True)
    
    output_path = Path('data/raw/XAUUSD/M1.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_m1.to_csv(output_path, index=True)
    
    print(f"✅ M1: {len(df_m1)} bars saved")
    print(f"   Period: {df_m1['time'].iloc[0]} to {df_m1['time'].iloc[-1]}")
else:
    print("❌ M1: No data")

mt5.shutdown()
print("\n✅ Export complete!")

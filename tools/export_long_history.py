"""
Export long history from MT5 (6+ years).
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    if not mt5.initialize():
        print("Failed to init MT5")
        return
    
    print("=" * 60)
    print("Exporting XAUUSD M5 + H1 (6 years)")
    print("=" * 60)
    
    # M5 - по годам
    print("\n📊 M5 Data:")
    m5_data = []
    for year in range(2019, 2026):
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1) if year < 2025 else datetime(2025, 12, 5)
        
        rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            m5_data.append(df)
            print(f"   {year}: {len(df):,} bars")
        else:
            print(f"   {year}: no data")
    
    if m5_data:
        m5_full = pd.concat(m5_data, ignore_index=True)
        m5_full['time'] = pd.to_datetime(m5_full['time'], unit='s')
        m5_full = m5_full.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        
        Path("data/raw/XAUUSD").mkdir(parents=True, exist_ok=True)
        m5_full.to_csv("data/raw/XAUUSD/M5.csv", index=False)
        print(f"\n   ✅ Total M5: {len(m5_full):,} bars")
        print(f"   Period: {m5_full['time'].iloc[0]} to {m5_full['time'].iloc[-1]}")
    
    # H1 - одним запросом
    print("\n📊 H1 Data:")
    h1_rates = mt5.copy_rates_range(
        "XAUUSD", 
        mt5.TIMEFRAME_H1, 
        datetime(2019, 1, 1), 
        datetime(2025, 12, 5)
    )
    
    if h1_rates is not None and len(h1_rates) > 0:
        h1_full = pd.DataFrame(h1_rates)
        h1_full['time'] = pd.to_datetime(h1_full['time'], unit='s')
        h1_full.to_csv("data/raw/XAUUSD/H1.csv", index=False)
        print(f"   ✅ Total H1: {len(h1_full):,} bars")
        print(f"   Period: {h1_full['time'].iloc[0]} to {h1_full['time'].iloc[-1]}")
    
    mt5.shutdown()
    print("\n✅ Export complete!")

if __name__ == "__main__":
    main()

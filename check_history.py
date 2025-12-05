"""Check available history in MT5"""
import MetaTrader5 as mt5
import pandas as pd

mt5.initialize()

symbol = 'XAUUSD'
timeframes = [
    ('M1', mt5.TIMEFRAME_M1),
    ('M5', mt5.TIMEFRAME_M5),
    ('M15', mt5.TIMEFRAME_M15),
    ('M30', mt5.TIMEFRAME_M30),
    ('H1', mt5.TIMEFRAME_H1),
    ('H4', mt5.TIMEFRAME_H4),
    ('D1', mt5.TIMEFRAME_D1),
    ('W1', mt5.TIMEFRAME_W1),
]

print("=" * 60)
print("AVAILABLE HISTORY FOR XAUUSD")
print("=" * 60)

for name, tf in timeframes:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 999999)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        start = df['time'].min()
        end = df['time'].max()
        years = (end - start).days / 365
        print(f"{name:4s}: {len(rates):>10,} bars | {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')} ({years:.1f} years)")
    else:
        print(f"{name:4s}: No data")

mt5.shutdown()

"""
Download historical XAUUSD data from free sources.

Sources:
1. Dukascopy (Swiss bank) - free tick/M1/M5 data
2. HistData.com - free M1 data
3. TrueFX - free tick data
"""

import os
import requests
import pandas as pd
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
import time

def download_dukascopy_data(symbol="XAUUSD", year=2024, month=1, timeframe="m5"):
    """
    Download data from Dukascopy (best free source).
    
    Timeframes: tick, m1, m5, m15, m30, h1, h4, d1
    """
    base_url = "https://datafeed.dukascopy.com/datafeed"
    
    # Dukascopy symbol mapping
    symbol_map = {
        "XAUUSD": "XAUUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
    }
    
    duka_symbol = symbol_map.get(symbol, symbol)
    
    # URL structure: https://datafeed.dukascopy.com/datafeed/XAUUSD/2024/0/01/BID_candles_min_5.bi5
    all_data = []
    
    print(f"\n📥 Downloading {symbol} {timeframe.upper()} for {year}...")
    
    for month_idx in range(12):
        print(f"   Month {month_idx + 1}/12...", end=" ")
        
        try:
            # Dukascopy uses 0-indexed months
            url = f"{base_url}/{duka_symbol}/{year}/{month_idx:02d}/01/BID_candles_min_5.bi5"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse binary data (Dukascopy format)
                # This is complex, need to use lzma decompression
                print("✓")
                # TODO: implement parser
            else:
                print("✗")
        
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.5)  # Be nice to server
    
    return None


def download_histdata_m1(symbol="XAUUSD", year=2024, month=1):
    """
    Download M1 data from HistData.com (easier format).
    
    Format: CSV in ZIP archive
    URL: http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd/2024/1
    """
    symbol_lower = symbol.lower()
    
    # HistData URL
    url = f"http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{symbol_lower}/{year}/{month}"
    
    print(f"\n📥 Downloading {symbol} M1 {year}-{month:02d} from HistData...")
    
    try:
        response = requests.get(url, timeout=30, allow_redirects=True)
        
        if response.status_code == 200 and len(response.content) > 1000:
            # Try to extract ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, names=['time', 'open', 'high', 'low', 'close', 'volume'])
                    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S')
                    print(f"   ✅ Got {len(df)} bars")
                    return df
        else:
            print(f"   ❌ Failed (status {response.status_code})")
            return None
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def download_investing_data(symbol="XAU/USD", from_date="01/01/2019", to_date="12/04/2025"):
    """
    Alternative: Investing.com historical data.
    Note: Requires manual download, but provides CSV.
    """
    print("\n💡 Manual alternative:")
    print("1. Go to: https://www.investing.com/currencies/xau-usd-historical-data")
    print("2. Set date range")
    print("3. Click 'Download' button")
    print("4. Save to: data/raw/XAUUSD/investing_h1.csv")


def download_yahoo_finance(symbol="GC=F", period="10y"):
    """
    Yahoo Finance - free gold futures data.
    Symbol: GC=F (Gold Futures)
    """
    try:
        import yfinance as yf
        
        print(f"\n📥 Downloading from Yahoo Finance ({symbol})...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1h")
        
        if not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'date': 'time', 'datetime': 'time'})
            print(f"   ✅ Got {len(df)} bars")
            print(f"   Period: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            return df
        else:
            print("   ❌ No data")
            return None
    
    except ImportError:
        print("   ⚠️  Need: pip install yfinance")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def main():
    """Download data from multiple sources."""
    
    print("=" * 60)
    print("Golden Breeze - Historical Data Downloader")
    print("=" * 60)
    
    output_dir = Path("data/raw/XAUUSD")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Yahoo Finance (easiest, H1 data)
    print("\n🎯 Trying Yahoo Finance (Gold Futures H1)...")
    df_yahoo = download_yahoo_finance("GC=F", period="10y")
    
    if df_yahoo is not None:
        df_yahoo.to_csv(output_dir / "yahoo_h1.csv", index=False)
        print(f"   💾 Saved to: {output_dir / 'yahoo_h1.csv'}")
        
        # Convert to M5 by resampling
        df_yahoo['time'] = pd.to_datetime(df_yahoo['time'])
        df_yahoo = df_yahoo.set_index('time')
        
        df_m5 = df_yahoo.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_m5 = df_m5.reset_index()
        df_m5.to_csv(output_dir / "yahoo_m5.csv", index=False)
        print(f"   💾 Resampled to M5: {len(df_m5)} bars")
        print(f"   💾 Saved to: {output_dir / 'yahoo_m5.csv'}")
    
    # Show manual alternatives
    print("\n" + "=" * 60)
    print("📌 Alternative sources (manual download):")
    print("=" * 60)
    print("\n1. Investing.com:")
    print("   https://www.investing.com/currencies/xau-usd-historical-data")
    print("   - Free, no registration")
    print("   - Daily/Hourly data")
    print("   - Manual CSV download")
    
    print("\n2. Dukascopy:")
    print("   https://www.dukascopy.com/swiss/english/marketwatch/historical/")
    print("   - Free, requires registration")
    print("   - M1/M5/Tick data")
    print("   - Manual download or API")
    
    print("\n3. HistData.com:")
    print("   http://www.histdata.com/download-free-forex-data/")
    print("   - Free M1 data")
    print("   - Limited symbols (mainly forex)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

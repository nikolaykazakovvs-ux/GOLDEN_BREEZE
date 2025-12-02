"""
Export historical data from MetaTrader 5 to CSV/Parquet files.

Usage:
    python -m tools.export_mt5_history --symbol XAUUSD --start 2024-01-01 --end 2024-06-01 --timeframes M1 M5 M15 H1 H4

Author: Golden Breeze Team
Version: 1.1
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import MetaTrader5 as mt5

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from mcp_servers.trading.market_data import get_ohlcv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export MT5 historical data to CSV/Parquet"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol (e.g., XAUUSD, XAUUSD.x)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["M1", "M5", "M15", "H1", "H4"],
        help="List of timeframes to export"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for exported data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output file format"
    )
    
    return parser.parse_args()


def timeframe_to_mt5(tf_str: str) -> int:
    """Convert timeframe string to MT5 constant."""
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }
    return mapping.get(tf_str, mt5.TIMEFRAME_M5)


def export_timeframe(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    output_path: Path,
    file_format: str = "csv"
) -> bool:
    """
    Export single timeframe data to file.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string (M1, M5, etc.)
        start_date: Start date
        end_date: End date
        output_path: Output file path
        file_format: File format (csv or parquet)
    
    Returns:
        True if successful
    """
    print(f"\n📊 Exporting {symbol} {timeframe} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # Вызываем get_ohlcv (передаём строку "M5", не MT5 константу)
        df = get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,  # передаём строку напрямую
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            count=100000  # достаточно для любого периода
        )
        
        if df is None or df.empty:
            print(f"❌ No data received for {timeframe}")
            return False
        
        # Создаём директорию если нужно
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в файл
        if file_format == "csv":
            df.to_csv(output_path)
        else:
            df.to_parquet(output_path)
        
        print(f"✅ Saved {len(df)} bars to {output_path}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Columns: {list(df.columns)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error exporting {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main export function."""
    args = parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"❌ Invalid date format: {e}")
        print("Use format: YYYY-MM-DD")
        sys.exit(1)
    
    # Validate dates
    if start_date >= end_date:
        print("❌ Start date must be before end date")
        sys.exit(1)
    
    print("="*60)
    print("Golden Breeze - MT5 Historical Data Export")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Timeframes: {', '.join(args.timeframes)}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print("="*60)
    
    # Проверяем подключение к MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        print(f"Error: {mt5.last_error()}")
        sys.exit(1)
    
    print(f"✅ Connected to MT5 (build {mt5.version()[0]})")
    
    # Export each timeframe
    output_base = Path(args.output_dir) / args.symbol
    success_count = 0
    total_count = len(args.timeframes)
    
    for tf in args.timeframes:
        extension = "csv" if args.format == "csv" else "parquet"
        output_path = output_base / f"{tf}.{extension}"
        
        if export_timeframe(
            symbol=args.symbol,
            timeframe=tf,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path,
            file_format=args.format
        ):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Export Summary: {success_count}/{total_count} timeframes exported")
    print("="*60)
    
    # Shutdown MT5
    mt5.shutdown()
    
    if success_count == total_count:
        print("✅ All exports completed successfully")
        sys.exit(0)
    else:
        print(f"⚠️  {total_count - success_count} exports failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Fast Label Generation - Simplified version without full backtest.

Instead of running full HybridStrategy backtest, directly compute labels
from price action and indicators. Much faster for smoke testing.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe."""
    df = df.copy()
    
    # Returns
    df['return'] = df['close'].pct_change()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # ATR
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift())
    df['lc'] = abs(df['low'] - df['close'].shift())
    df['atr'] = df[['hl', 'hc', 'lc']].max(axis=1).rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    
    # Volume ratio (use tick_volume if volume not available)
    if 'tick_volume' in df.columns:
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
    elif 'real_volume' in df.columns:
        df['volume_ratio'] = df['real_volume'] / df['real_volume'].rolling(20).mean()
    elif 'volume' in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    else:
        df['volume_ratio'] = 1.0
    
    return df


def generate_direction_labels(df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
    """
    Generate direction labels based on future price movement.
    
    Args:
        df: DataFrame with OHLCV + indicators
        lookahead: Bars to look ahead for target
        
    Returns:
        DataFrame with 'direction_label' column:
            0 = FLAT (low movement)
            1 = LONG (upward movement)
            2 = SHORT (downward movement)
    """
    df = df.copy()
    
    # Calculate future return
    df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
    
    # Calculate threshold based on ATR
    # Movement > 0.5 * ATR = directional, otherwise FLAT
    threshold = 0.5 * df['atr'] / df['close']
    
    # Generate labels
    conditions = [
        df['future_return'] > threshold,   # LONG
        df['future_return'] < -threshold,  # SHORT
    ]
    choices = [1, 2]  # LONG=1, SHORT=2
    df['direction_label'] = np.select(conditions, choices, default=0)  # FLAT=0
    
    # Drop rows with NaN (not enough history or future data)
    df = df.dropna()
    
    return df


def load_timeframe_data(data_dir: Path, symbol: str) -> Dict[str, pd.DataFrame]:
    """Load all timeframe data and add indicators."""
    timeframes = ["M1", "M5", "M15", "H1", "H4"]
    data = {}
    
    for tf in timeframes:
        csv_path = data_dir / symbol / f"{tf}.csv"
        if not csv_path.exists():
            print(f"⚠️  {tf} data not found: {csv_path}")
            continue
        
        # Read CSV and find timestamp column
        df = pd.read_csv(csv_path)
        
        # Find timestamp column ('time', 'timestamp', or first column)
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        
        if time_col:
            # Parse timestamps and set as index
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index(time_col)
        else:
            print(f"⚠️  No time column found in {csv_path}, skipping")
            continue
        
        print(f"✅ Loaded {tf}: {len(df)} bars from {csv_path}")
        
        # Add indicators
        df = add_indicators(df)
        data[tf] = df
        
    return data


def generate_labels_from_data(
    primary_df: pd.DataFrame,
    symbol: str,
    primary_tf: str,
    lookahead: int = 10
) -> pd.DataFrame:
    """
    Generate training labels from primary timeframe data.
    
    Args:
        primary_df: Primary timeframe OHLCV data with indicators
        symbol: Trading symbol
        primary_tf: Primary timeframe
        lookahead: Bars to look ahead
        
    Returns:
        DataFrame with labels
    """
    print("\n" + "="*60)
    print(f"Generating labels for {symbol} on {primary_tf}")
    print(f"Lookahead: {lookahead} bars")
    print("="*60)
    
    # Generate labels
    labeled_df = generate_direction_labels(primary_df, lookahead=lookahead)
    
    # Count labels
    label_counts = labeled_df['direction_label'].value_counts().sort_index()
    total = len(labeled_df)
    
    print(f"\n✅ Generated {total} labeled samples")
    print("\nLabel Distribution:")
    print(f"  FLAT (0):  {label_counts.get(0, 0):5d} ({100*label_counts.get(0, 0)/total:5.1f}%)")
    print(f"  LONG (1):  {label_counts.get(1, 0):5d} ({100*label_counts.get(1, 0)/total:5.1f}%)")
    print(f"  SHORT (2): {label_counts.get(2, 0):5d} ({100*label_counts.get(2, 0)/total:5.1f}%)")
    
    # Prepare output - keep timestamp as column (not index)
    output_df = labeled_df[['direction_label']].copy()
    # Convert index to datetime string (ISO format) for CSV storage
    output_df['timestamp'] = pd.to_datetime(labeled_df.index).strftime('%Y-%m-%d %H:%M:%S%z')
    output_df['symbol'] = symbol
    output_df['timeframe'] = primary_tf
    
    return output_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fast label generation for Direction LSTM"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., XAUUSD)"
    )
    parser.add_argument(
        "--primary-tf",
        type=str,
        default="M5",
        help="Primary timeframe (default: M5)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Data directory (default: data/raw)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=10,
        help="Bars to look ahead for target (default: 10)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("Golden Breeze - Fast Label Generation")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Primary TF: {args.primary_tf}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Lookahead: {args.lookahead}")
    print("="*60)
    
    # Load data
    multitf_data = load_timeframe_data(args.data_dir, args.symbol)
    
    if args.primary_tf not in multitf_data:
        print(f"❌ Primary timeframe {args.primary_tf} not found!")
        sys.exit(1)
    
    primary_df = multitf_data[args.primary_tf]
    
    # Generate labels
    labels_df = generate_labels_from_data(
        primary_df=primary_df,
        symbol=args.symbol,
        primary_tf=args.primary_tf,
        lookahead=args.lookahead
    )
    
    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(args.output)
    print(f"\n✅ Labels saved to: {args.output}")
    print(f"   Rows: {len(labels_df)}")
    print(f"   Columns: {list(labels_df.columns)}")


if __name__ == "__main__":
    main()

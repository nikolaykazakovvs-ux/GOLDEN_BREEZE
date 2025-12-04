"""
Prepare dataset for Direction LSTM training from labels.

Converts labeled trades + OHLCV data into sequences suitable for LSTM training.

Usage:
    python -m aimodule.training.prepare_direction_dataset \
        --labels data/labels/direction_labels.csv \
        --data-dir data/raw \
        --symbol XAUUSD \
        --timeframe M5 \
        --seq-len 50 \
        --output data/prepared/direction_dataset.npz

Author: Golden Breeze Team
Version: 1.1
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Import feature engineering pipeline
from aimodule.data_pipeline.features import add_basic_features
from aimodule.data_pipeline.features_gold import GOLD_FEATURE_COLUMNS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Direction LSTM dataset from labels"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels CSV file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory with raw OHLCV data"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="M5",
        help="Timeframe for sequences"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Sequence length for LSTM"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/prepared/direction_dataset.npz",
        help="Output file for prepared dataset"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set fraction (0.0-1.0)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation set fraction from train (0.0-1.0)"
    )
    
    return parser.parse_args()


def load_ohlcv_data(data_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data for given symbol and timeframe."""
    symbol_dir = data_dir / symbol
    
    # Пробуем CSV и Parquet
    csv_path = symbol_dir / f"{timeframe}.csv"
    parquet_path = symbol_dir / f"{timeframe}.parquet"
    
    if csv_path.exists():
        # Read without setting index first
        df = pd.read_csv(csv_path)
        
        # Find timestamp column ('time', 'timestamp', or first column)
        time_col = None
        if 'time' in df.columns:
            time_col = 'time'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            # Try first column if it looks like timestamps
            first_col = df.columns[0]
            if df[first_col].dtype == 'object':
                time_col = first_col
        
        if time_col:
            # Parse timestamps and set as index
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.set_index(time_col)
        else:
            raise ValueError(f"No timestamp column found in {csv_path}")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
    else:
        raise FileNotFoundError(f"Data not found: {csv_path} or {parquet_path}")
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features using unified pipeline."""
    df = df.copy()
    
    # Use integrated feature pipeline (includes SMC + Gold features)
    df = add_basic_features(df, use_gold_features=True)
    
    # Add additional features specific to this dataset
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # SMA ratio
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        df['sma_ratio'] = df['sma_fast'] / df['sma_slow']
    
    # ATR normalized
    if 'atr' in df.columns:
        df['atr_norm'] = df['atr'] / df['close']
    
    # RSI (additional for backward compatibility)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std_val = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std_val * bb_std)
    df['bb_lower'] = df['bb_mid'] - (bb_std_val * bb_std)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features (if available)
    if 'tick_volume' in df.columns:
        df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
    elif 'real_volume' in df.columns:
        df['volume_sma'] = df['real_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['real_volume'] / df['volume_sma']
    else:
        df['volume_ratio'] = 1.0
    
    # Drop NaN values
    df = df.dropna()
    
    return df


def create_sequences(
    df: pd.DataFrame,
    labels_df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        df: OHLCV DataFrame with features
        labels_df: Labels DataFrame with timestamps
        seq_len: Sequence length
        feature_cols: List of feature column names
    
    Returns:
        (X, y, timestamps) arrays
    """
    # Merge labels with OHLCV data by timestamp
    df = df.copy()
    df['timestamp'] = df.index
    
    # Create label column in main df (default to 0=FLAT)
    df['label'] = 0
    
    # Assign labels from labels_df
    for _, row in labels_df.iterrows():
        ts = pd.to_datetime(row['timestamp'])
        # Find closest timestamp in df
        idx = df.index.searchsorted(ts)
        if idx < len(df):
            df.iloc[idx, df.columns.get_loc('label')] = row['direction_label']
    
    # Prepare sequences
    X_list = []
    y_list = []
    ts_list = []
    
    for i in range(seq_len, len(df)):
        # Get sequence window
        seq = df.iloc[i-seq_len:i][feature_cols].values
        
        # Get label at current timestamp
        label = df.iloc[i]['label']
        
        # Only include if label is not FLAT (0) - we want to predict actual trades
        if label > 0:
            X_list.append(seq)
            y_list.append(label)
            ts_list.append(df.index[i])
    
    if not X_list:
        raise ValueError("No valid sequences created. Check labels and data alignment.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    timestamps = np.array(ts_list)
    
    return X, y, timestamps


def prepare_dataset(
    labels_path: Path,
    data_dir: Path,
    symbol: str,
    timeframe: str,
    seq_len: int,
    test_split: float,
    val_split: float
) -> dict:
    """
    Prepare complete dataset for LSTM training.
    
    Returns:
        Dictionary with train/val/test splits and metadata
    """
    print("="*60)
    print("Preparing Direction LSTM Dataset")
    print("="*60)
    
    # Load labels
    print(f"Loading labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    # Handle Unix timestamps (seconds) or datetime strings
    try:
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'], unit='s', utc=True)
    except:
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
    print(f"✅ Loaded {len(labels_df)} labels")
    print(f"   Label distribution: {labels_df['direction_label'].value_counts().to_dict()}")
    
    # Load OHLCV data
    print(f"\nLoading OHLCV data ({symbol} {timeframe})...")
    df = load_ohlcv_data(data_dir, symbol, timeframe)
    print(f"✅ Loaded {len(df)} bars")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Add features
    print("\nAdding technical features...")
    df = add_features(df)
    print(f"✅ Added features, {len(df)} bars after dropna")
    
    # Feature columns (now includes SMC + Gold features)
    base_feature_cols = [
        'close', 'returns', 'log_returns',
        'sma_fast', 'sma_slow', 'sma_ratio',
        'atr', 'atr_norm',
        'rsi',
        'bb_position',
        'volume_ratio',
        # SMC features
        'SMC_FVG_Bullish', 'SMC_FVG_Bearish',
        'SMC_Swing_High', 'SMC_Swing_Low'
    ]
    
    # Add Gold features if available in dataframe
    gold_features_in_df = [col for col in GOLD_FEATURE_COLUMNS if col in df.columns]
    feature_cols = base_feature_cols + gold_features_in_df
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(f"  Base features: {len(base_feature_cols)}")
    print(f"  Gold features: {len(gold_features_in_df)}")
    if gold_features_in_df:
        print(f"  Gold features added: {', '.join(gold_features_in_df[:5])}{'...' if len(gold_features_in_df) > 5 else ''}")
    print(f"  Total: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    # Create sequences
    print(f"\nCreating sequences (seq_len={seq_len})...")
    X, y, timestamps = create_sequences(df, labels_df, seq_len, feature_cols)
    print(f"✅ Created {len(X)} sequences")
    print(f"   Shape: X={X.shape}, y={y.shape}")
    
    # Normalize features (per-feature across all sequences)
    print("\nNormalizing features...")
    n_samples, seq_len_actual, n_features = X.shape
    
    # Reshape for scaling: (samples * seq_len, features)
    X_reshaped = X.reshape(-1, n_features)
    
    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape back
    X_scaled = X_scaled.reshape(n_samples, seq_len_actual, n_features)
    
    print(f"✅ Normalized with StandardScaler")
    print(f"   Mean: {scaler.mean_[:5]}")
    print(f"   Std: {scaler.scale_[:5]}")
    
    # Split into train/test
    print(f"\nSplitting data (test={test_split:.1%})...")
    X_train_full, X_test, y_train_full, y_test, ts_train_full, ts_test = train_test_split(
        X_scaled, y, timestamps,
        test_size=test_split,
        random_state=42,
        stratify=y
    )
    
    # Split train into train/val
    print(f"Splitting train into train/val (val={val_split:.1%})...")
    X_train, X_val, y_train, y_val, ts_train, ts_val = train_test_split(
        X_train_full, y_train_full, ts_train_full,
        test_size=val_split,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"\n✅ Dataset split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Class distribution
    print(f"\nClass distribution:")
    for split_name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(split_y, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"   {split_name}: {dist}")
    
    # Prepare output
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'timestamps_train': ts_train,
        'X_val': X_val,
        'y_val': y_val,
        'timestamps_val': ts_val,
        'X_test': X_test,
        'y_test': y_test,
        'timestamps_test': ts_test,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'feature_cols': feature_cols,
        'seq_len': seq_len,
        'n_features': n_features,
        'n_classes': len(np.unique(y)),
        'symbol': symbol,
        'timeframe': timeframe,
    }
    
    return dataset


def main():
    """Main preparation function."""
    args = parse_args()
    
    print("="*60)
    print("Golden Breeze - Direction LSTM Dataset Preparation")
    print("="*60)
    print(f"Labels: {args.labels}")
    print(f"Data dir: {args.data_dir}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Test split: {args.test_split:.1%}")
    print(f"Val split: {args.val_split:.1%}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Prepare dataset
    try:
        dataset = prepare_dataset(
            labels_path=Path(args.labels),
            data_dir=Path(args.data_dir),
            symbol=args.symbol,
            timeframe=args.timeframe,
            seq_len=args.seq_len,
            test_split=args.test_split,
            val_split=args.val_split
        )
    except Exception as e:
        print(f"\n❌ Error preparing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving dataset to {output_path}...")
    np.savez_compressed(output_path, **dataset)
    
    print(f"✅ Dataset saved successfully")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*60)
    print("Dataset preparation completed successfully!")
    print("="*60)
    
    print(f"\nNext step:")
    print(f"  python -m aimodule.training.train_direction_lstm_from_labels \\")
    print(f"    --data {args.output} \\")
    print(f"    --seq-len {args.seq_len} \\")
    print(f"    --epochs 20 \\")
    print(f"    --save-path models/direction_lstm_hybrid.pt")


if __name__ == "__main__":
    main()

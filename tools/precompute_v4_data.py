"""
V4 Data Pre-computation Factory

Creates optimized dataset for V4 5-class training.
Combines: M5 OHLCV + H1 OHLCV + SMC + Strategy Signals + 5-Class Labels

Output: data/prepared/v4_5class_dataset.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from aimodule.training.labeling_v5 import LabelingV5, generate_5class_labels
from aimodule.data_pipeline.strategy_signals import StrategySignalsGenerator
from aimodule.data_pipeline.features_v3 import V3Features


class SMCProcessor:
    """
    Smart Money Concepts processor for H1 data.
    Generates: Order Blocks, Fair Value Gaps, Market Structure.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def process(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate SMC features from H1 data.
        
        Returns:
            np.ndarray of shape (n_samples, 8) with SMC features
        """
        n = len(df)
        smc = np.zeros((n, 8), dtype=np.float32)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        
        for i in range(self.lookback, n):
            # 1. Price position in range
            window_high = high[i-self.lookback:i+1].max()
            window_low = low[i-self.lookback:i+1].min()
            range_val = window_high - window_low
            
            if range_val > 0:
                smc[i, 0] = (close[i] - window_low) / range_val  # Position 0-1
                smc[i, 1] = (window_high - close[i]) / range_val  # Distance to high
            
            # 2. Candle structure
            body = abs(close[i] - open_[i])
            wick_up = high[i] - max(close[i], open_[i])
            wick_down = min(close[i], open_[i]) - low[i]
            total_range = high[i] - low[i]
            
            if total_range > 0:
                smc[i, 2] = body / total_range  # Body ratio
                smc[i, 3] = wick_up / total_range  # Upper wick ratio
                smc[i, 4] = wick_down / total_range  # Lower wick ratio
            
            # 3. Market structure
            # Higher high / Lower low detection
            recent_highs = high[i-5:i]
            recent_lows = low[i-5:i]
            
            if len(recent_highs) >= 5:
                smc[i, 5] = 1 if high[i] > recent_highs.max() else (-1 if high[i] < recent_highs.min() else 0)
                smc[i, 6] = 1 if low[i] > recent_lows.max() else (-1 if low[i] < recent_lows.min() else 0)
            
            # 4. Momentum
            if i >= 10:
                momentum = (close[i] - close[i-10]) / close[i-10]
                smc[i, 7] = np.clip(momentum * 10, -1, 1)  # Normalized
        
        return smc


class TimeAligner:
    """
    Aligns M5 and H1 data by timestamp.
    """
    
    def __init__(self):
        pass
    
    def align(
        self, 
        df_m5: pd.DataFrame, 
        df_h1: pd.DataFrame,
    ) -> tuple:
        """
        Align M5 data with corresponding H1 bars.
        
        Returns:
            (m5_indices, h1_indices) - aligned index arrays
        """
        # Ensure datetime
        if 'time' in df_m5.columns:
            m5_times = pd.to_datetime(df_m5['time'])
        elif 'timestamp' in df_m5.columns:
            m5_times = pd.to_datetime(df_m5['timestamp'])
        else:
            m5_times = df_m5.index
            
        if 'time' in df_h1.columns:
            h1_times = pd.to_datetime(df_h1['time'])
        elif 'timestamp' in df_h1.columns:
            h1_times = pd.to_datetime(df_h1['timestamp'])
        else:
            h1_times = df_h1.index
        
        # Floor M5 to hour
        m5_hours = m5_times.dt.floor('h')
        
        # Create mapping
        h1_time_to_idx = {t: i for i, t in enumerate(h1_times)}
        
        m5_indices = []
        h1_indices = []
        
        for i, m5_hour in enumerate(m5_hours):
            if m5_hour in h1_time_to_idx:
                m5_indices.append(i)
                h1_indices.append(h1_time_to_idx[m5_hour])
        
        return np.array(m5_indices), np.array(h1_indices)


def precompute_v4_dataset(
    m5_path: str = "data/raw/XAUUSD/M5.csv",
    h1_path: str = "data/raw/XAUUSD/H1.csv",
    output_path: str = "data/prepared/v4_5class_dataset.npz",
    horizon: int = 12,
    strong_thresh: float = 0.004,
    weak_thresh: float = 0.001,
    seq_len_fast: int = 50,
    seq_len_slow: int = 20,
):
    """
    Pre-compute V4 5-class dataset.
    
    Pipeline:
    1. Load M5, H1 data
    2. Generate 5-class labels (M5)
    3. Generate strategy signals (M5)
    4. Process SMC features (H1)
    5. Align M5 <-> H1
    6. Create sequences
    7. Save to .npz
    """
    
    print("=" * 70)
    print("V4 5-Class Dataset Pre-computation")
    print("=" * 70)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n📂 Step 1: Loading data...")
    
    df_m5 = pd.read_csv(m5_path)
    df_h1 = pd.read_csv(h1_path)
    
    print(f"   M5: {len(df_m5):,} bars")
    print(f"   H1: {len(df_h1):,} bars")
    
    # =========================================================================
    # 2. GENERATE 5-CLASS LABELS
    # =========================================================================
    print("\n🏷️ Step 2: Generating 5-class labels...")
    
    labeler = LabelingV5(
        horizon=horizon,
        strong_thresh=strong_thresh,
        weak_thresh=weak_thresh,
    )
    
    labels_df = labeler.generate_labels(df_m5)
    labels = labels_df['label'].values
    
    distribution = labeler.get_class_distribution(labels)
    print(f"   Distribution:")
    for cls_name, stats in distribution.items():
        print(f"      {cls_name}: {stats['count']:,} ({stats['percent']:.1f}%)")
    
    class_weights = labeler.get_class_weights(labels)
    
    # =========================================================================
    # 3. GENERATE STRATEGY SIGNALS
    # =========================================================================
    print("\n📊 Step 3: Generating strategy signals...")
    
    signal_gen = StrategySignalsGenerator()
    signals_df = signal_gen.generate_all_signals(df_m5)
    signals = signals_df.values.astype(np.float32)
    
    print(f"   Signals shape: {signals.shape}")
    print(f"   Features: {signal_gen.get_feature_dim()}")
    
    # =========================================================================
    # 4. GENERATE V3 FEATURES (for M5)
    # =========================================================================
    print("\n🔧 Step 4: Generating V3 features...")
    
    v3_gen = V3Features()
    v3_df = v3_gen.extract_features(df_m5)
    v3_features = v3_df.values.astype(np.float32)
    
    print(f"   V3 features shape: {v3_features.shape}")
    
    # =========================================================================
    # 5. PROCESS SMC (H1)
    # =========================================================================
    print("\n🎯 Step 5: Processing SMC features from H1...")
    
    smc_processor = SMCProcessor(lookback=50)
    smc_h1 = smc_processor.process(df_h1)
    
    print(f"   SMC shape: {smc_h1.shape}")
    
    # =========================================================================
    # 6. ALIGN M5 <-> H1
    # =========================================================================
    print("\n⏱️ Step 6: Aligning M5 and H1...")
    
    aligner = TimeAligner()
    m5_idx, h1_idx = aligner.align(df_m5, df_h1)
    
    print(f"   Aligned pairs: {len(m5_idx):,}")
    
    # =========================================================================
    # 7. CREATE SEQUENCES
    # =========================================================================
    print("\n📦 Step 7: Creating sequences...")
    
    # Valid indices (have enough history and valid labels)
    min_idx = max(seq_len_fast, seq_len_slow * 12)  # H1 needs 12x for M5 alignment
    
    valid_mask = (
        (labels >= 0) & 
        (np.arange(len(labels)) >= min_idx) &
        (np.arange(len(labels)) < len(labels) - horizon)
    )
    
    # Filter to aligned indices only
    aligned_set = set(m5_idx)
    valid_mask = valid_mask & np.array([i in aligned_set for i in range(len(labels))])
    
    valid_indices = np.where(valid_mask)[0]
    print(f"   Valid samples: {len(valid_indices):,}")
    
    # Pre-allocate arrays
    n_samples = len(valid_indices)
    
    # M5 sequences (fast stream)
    x_fast = np.zeros((n_samples, seq_len_fast, v3_features.shape[1]), dtype=np.float32)
    
    # H1 data at each point (slow stream)
    x_slow = np.zeros((n_samples, seq_len_slow, smc_h1.shape[1]), dtype=np.float32)
    
    # Strategy signals (current bar)
    x_strategy = np.zeros((n_samples, signals.shape[1]), dtype=np.float32)
    
    # Labels
    y = np.zeros(n_samples, dtype=np.int64)
    
    # Create index mapping for H1
    m5_to_h1 = {m: h for m, h in zip(m5_idx, h1_idx)}
    
    print("   Building sequences...")
    for i, idx in enumerate(valid_indices):
        if i % 5000 == 0:
            print(f"      {i:,} / {n_samples:,}")
        
        # M5 sequence
        x_fast[i] = v3_features[idx - seq_len_fast:idx]
        
        # H1 sequence (need to find corresponding H1 bars)
        h1_current = m5_to_h1.get(idx, 0)
        h1_start = max(0, h1_current - seq_len_slow)
        h1_slice = smc_h1[h1_start:h1_current]
        
        if len(h1_slice) < seq_len_slow:
            # Pad with zeros at the beginning
            pad_len = seq_len_slow - len(h1_slice)
            x_slow[i, pad_len:] = h1_slice
        else:
            x_slow[i] = h1_slice[-seq_len_slow:]
        
        # Strategy signals
        x_strategy[i] = signals[idx]
        
        # Label
        y[i] = labels[idx]
    
    # =========================================================================
    # 8. SAVE
    # =========================================================================
    print("\n💾 Step 8: Saving dataset...")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        x_fast=x_fast,
        x_slow=x_slow,
        x_strategy=x_strategy,
        y=y,
        class_weights=class_weights,
        valid_indices=valid_indices,
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"   Saved to: {output_path}")
    print(f"   File size: {file_size:.1f} MB")
    
    # Save metadata
    metadata = {
        "created": datetime.now().isoformat(),
        "m5_path": m5_path,
        "h1_path": h1_path,
        "n_samples": int(n_samples),
        "seq_len_fast": seq_len_fast,
        "seq_len_slow": seq_len_slow,
        "horizon": horizon,
        "strong_thresh": strong_thresh,
        "weak_thresh": weak_thresh,
        "x_fast_shape": list(x_fast.shape),
        "x_slow_shape": list(x_slow.shape),
        "x_strategy_shape": list(x_strategy.shape),
        "n_classes": 5,
        "class_distribution": {k: v['count'] for k, v in distribution.items()},
        "class_weights": class_weights.tolist(),
        "strategy_features": signal_gen.get_signal_names(),
    }
    
    meta_path = output_path.replace('.npz', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {meta_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Dataset Pre-computation Complete!")
    print("=" * 70)
    print(f"""
Dataset Summary:
   Samples:        {n_samples:,}
   Fast (M5):      {x_fast.shape}
   Slow (H1):      {x_slow.shape}
   Strategy:       {x_strategy.shape}
   Labels:         {y.shape} (5 classes)
   
Class Distribution:
""")
    for cls_name, stats in distribution.items():
        print(f"   {cls_name}: {stats['count']:,} ({stats['percent']:.1f}%)")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-compute V4 5-class dataset")
    parser.add_argument("--m5", type=str, default="data/raw/XAUUSD/M5.csv",
                        help="Path to M5 CSV")
    parser.add_argument("--h1", type=str, default="data/raw/XAUUSD/H1.csv",
                        help="Path to H1 CSV")
    parser.add_argument("--output", type=str, default="data/prepared/v4_5class_dataset.npz",
                        help="Output path")
    parser.add_argument("--horizon", type=int, default=12,
                        help="Label horizon (bars)")
    parser.add_argument("--strong", type=float, default=0.004,
                        help="Strong threshold (0.4%)")
    parser.add_argument("--weak", type=float, default=0.001,
                        help="Weak threshold (0.1%)")
    parser.add_argument("--seq-fast", type=int, default=50,
                        help="M5 sequence length")
    parser.add_argument("--seq-slow", type=int, default=20,
                        help="H1 sequence length")
    
    args = parser.parse_args()
    
    precompute_v4_dataset(
        m5_path=args.m5,
        h1_path=args.h1,
        output_path=args.output,
        horizon=args.horizon,
        strong_thresh=args.strong,
        weak_thresh=args.weak,
        seq_len_fast=args.seq_fast,
        seq_len_slow=args.seq_slow,
    )

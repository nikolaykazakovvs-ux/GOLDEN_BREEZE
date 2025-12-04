"""
5-Class Labeling System for Golden Breeze V4

Classes:
    0: STRONG_DOWN  (movement < -strong_thresh)
    1: WEAK_DOWN    (movement -strong_thresh ... -weak_thresh)
    2: NEUTRAL      (movement -weak_thresh ... +weak_thresh)
    3: WEAK_UP      (movement +weak_thresh ... +strong_thresh)
    4: STRONG_UP    (movement > +strong_thresh)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


class LabelingV5:
    """5-Class labeling generator for trading signals."""
    
    # Class names for reference
    CLASS_NAMES = {
        0: "STRONG_DOWN",
        1: "WEAK_DOWN", 
        2: "NEUTRAL",
        3: "WEAK_UP",
        4: "STRONG_UP"
    }
    
    def __init__(
        self,
        horizon: int = 12,
        strong_thresh: float = 0.004,  # 0.4%
        weak_thresh: float = 0.001,    # 0.1%
    ):
        """
        Initialize 5-class labeler.
        
        Args:
            horizon: Number of bars to look ahead for return calculation
            strong_thresh: Threshold for STRONG moves (default 0.4%)
            weak_thresh: Threshold for WEAK moves (default 0.1%)
        """
        self.horizon = horizon
        self.strong_thresh = strong_thresh
        self.weak_thresh = weak_thresh
        
    def generate_labels(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Generate 5-class labels based on future returns.
        
        Args:
            df: DataFrame with OHLCV data
            close_col: Name of close price column
            
        Returns:
            DataFrame with 'label' (0-4) and 'future_return' columns
        """
        if close_col not in df.columns:
            raise ValueError(f"Column '{close_col}' not found in DataFrame")
            
        close = df[close_col].values
        n = len(close)
        
        # Calculate future returns
        future_returns = np.zeros(n, dtype=np.float32)
        labels = np.full(n, -1, dtype=np.int32)  # -1 = invalid
        
        for i in range(n - self.horizon):
            future_close = close[i + self.horizon]
            current_close = close[i]
            
            if current_close > 0:
                ret = (future_close - current_close) / current_close
                future_returns[i] = ret
                
                # Assign class
                if ret < -self.strong_thresh:
                    labels[i] = 0  # STRONG_DOWN
                elif ret < -self.weak_thresh:
                    labels[i] = 1  # WEAK_DOWN
                elif ret <= self.weak_thresh:
                    labels[i] = 2  # NEUTRAL
                elif ret <= self.strong_thresh:
                    labels[i] = 3  # WEAK_UP
                else:
                    labels[i] = 4  # STRONG_UP
        
        # Create result DataFrame
        result = pd.DataFrame({
            'label': labels,
            'future_return': future_returns,
        }, index=df.index)
        
        return result
    
    def get_class_distribution(self, labels: np.ndarray) -> dict:
        """Get distribution of classes."""
        valid_labels = labels[labels >= 0]
        unique, counts = np.unique(valid_labels, return_counts=True)
        
        distribution = {}
        total = len(valid_labels)
        
        for cls in range(5):
            count = counts[unique == cls][0] if cls in unique else 0
            pct = count / total * 100 if total > 0 else 0
            distribution[self.CLASS_NAMES[cls]] = {
                'count': int(count),
                'percent': round(pct, 2)
            }
            
        return distribution
    
    def get_class_weights(self, labels: np.ndarray) -> np.ndarray:
        """Calculate class weights for balanced training."""
        valid_labels = labels[labels >= 0]
        n_samples = len(valid_labels)
        n_classes = 5
        
        weights = np.ones(n_classes, dtype=np.float32)
        
        for cls in range(n_classes):
            count = np.sum(valid_labels == cls)
            if count > 0:
                weights[cls] = n_samples / (n_classes * count)
                
        return weights


def generate_5class_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    strong_thresh: float = 0.004,
    weak_thresh: float = 0.001,
    close_col: str = 'close',
) -> pd.DataFrame:
    """
    Convenience function to generate 5-class labels.
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Bars to look ahead (default 12 = 1 hour on M5)
        strong_thresh: Strong move threshold (default 0.4%)
        weak_thresh: Weak move threshold (default 0.1%)
        close_col: Close price column name
        
    Returns:
        DataFrame with 'label' and 'future_return' columns
    """
    labeler = LabelingV5(
        horizon=horizon,
        strong_thresh=strong_thresh,
        weak_thresh=weak_thresh,
    )
    return labeler.generate_labels(df, close_col)


def load_and_label(
    csv_path: str,
    horizon: int = 12,
    strong_thresh: float = 0.004,
    weak_thresh: float = 0.001,
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load CSV and generate 5-class labels.
    
    Args:
        csv_path: Path to OHLCV CSV file
        horizon: Bars to look ahead
        strong_thresh: Strong move threshold
        weak_thresh: Weak move threshold
        save_path: Optional path to save labels CSV
        
    Returns:
        Tuple of (labels DataFrame, class distribution dict)
    """
    print(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   {len(df)} bars loaded")
    
    labeler = LabelingV5(
        horizon=horizon,
        strong_thresh=strong_thresh,
        weak_thresh=weak_thresh,
    )
    
    print(f"🏷️ Generating 5-class labels...")
    print(f"   Horizon: {horizon} bars")
    print(f"   Strong threshold: ±{strong_thresh*100:.2f}%")
    print(f"   Weak threshold: ±{weak_thresh*100:.2f}%")
    
    labels_df = labeler.generate_labels(df)
    
    # Get distribution
    distribution = labeler.get_class_distribution(labels_df['label'].values)
    
    print(f"\n📊 Class Distribution:")
    for cls_name, stats in distribution.items():
        print(f"   {cls_name}: {stats['count']:,} ({stats['percent']:.1f}%)")
    
    # Calculate class weights
    weights = labeler.get_class_weights(labels_df['label'].values)
    print(f"\n⚖️ Class Weights:")
    for i, (cls_name, w) in enumerate(zip(LabelingV5.CLASS_NAMES.values(), weights)):
        print(f"   {cls_name}: {w:.3f}")
    
    # Save if requested
    if save_path:
        # Add timestamp from original data if available
        if 'time' in df.columns:
            labels_df['time'] = df['time']
        elif 'timestamp' in df.columns:
            labels_df['time'] = df['timestamp']
            
        labels_df.to_csv(save_path, index=False)
        print(f"\n💾 Saved to {save_path}")
    
    return labels_df, distribution


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 5-class labels for trading")
    parser.add_argument("--input", type=str, default="data/raw/XAUUSD/M5.csv",
                        help="Input OHLCV CSV file")
    parser.add_argument("--output", type=str, default="data/labels/labels_5class_XAUUSD.csv",
                        help="Output labels CSV file")
    parser.add_argument("--horizon", type=int, default=12,
                        help="Bars to look ahead (default 12)")
    parser.add_argument("--strong", type=float, default=0.004,
                        help="Strong threshold (default 0.004 = 0.4%)")
    parser.add_argument("--weak", type=float, default=0.001,
                        help="Weak threshold (default 0.001 = 0.1%)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("5-Class Label Generator")
    print("=" * 60)
    
    labels_df, dist = load_and_label(
        csv_path=args.input,
        horizon=args.horizon,
        strong_thresh=args.strong,
        weak_thresh=args.weak,
        save_path=args.output,
    )
    
    # Summary
    valid = labels_df['label'] >= 0
    print(f"\n✅ Generated {valid.sum():,} valid labels")
    print(f"   Invalid (end of data): {(~valid).sum():,}")

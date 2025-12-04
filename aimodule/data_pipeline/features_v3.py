"""
Feature Engineering for v4 Lite

Creates the same 15 engineered features that v3 uses:
- close, returns, log_returns
- sma_fast, sma_slow, sma_ratio
- atr, atr_norm
- rsi, bb_position
- volume_ratio
- SMC_FVG_Bullish, SMC_FVG_Bearish, SMC_Swing_High, SMC_Swing_Low

This makes v4 Lite use the same successful feature set as v3.

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class V3Features:
    """
    Feature engineering that exactly matches v3 LSTM model.
    
    The v3 model (DirectionLSTMModel) uses these 15 features:
    1. close - Normalized close price
    2. returns - Simple returns
    3. log_returns - Log returns
    4. sma_fast - Fast SMA (10 periods)
    5. sma_slow - Slow SMA (50 periods)
    6. sma_ratio - sma_fast / sma_slow
    7. atr - Average True Range
    8. atr_norm - ATR normalized by close
    9. rsi - Relative Strength Index (14 periods)
    10. bb_position - Position within Bollinger Bands
    11. volume_ratio - Volume relative to moving average
    12. SMC_FVG_Bullish - Bullish Fair Value Gap
    13. SMC_FVG_Bearish - Bearish Fair Value Gap
    14. SMC_Swing_High - Swing High marker
    15. SMC_Swing_Low - Swing Low marker
    """
    
    FEATURE_NAMES = [
        'close', 'returns', 'log_returns',
        'sma_fast', 'sma_slow', 'sma_ratio',
        'atr', 'atr_norm',
        'rsi', 'bb_position',
        'volume_ratio',
        'SMC_FVG_Bullish', 'SMC_FVG_Bearish',
        'SMC_Swing_High', 'SMC_Swing_Low'
    ]
    
    def __init__(
        self,
        sma_fast_period: int = 10,
        sma_slow_period: int = 50,
        atr_period: int = 14,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_ma_period: int = 20,
        swing_lookback: int = 5,
    ):
        """
        Initialize feature generator with periods.
        
        All default periods match v3's DirectionLSTMModel.
        """
        self.sma_fast_period = sma_fast_period
        self.sma_slow_period = sma_slow_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_ma_period = volume_ma_period
        self.swing_lookback = swing_lookback
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all 15 features from OHLCV data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume (or tick_volume)
            
        Returns:
            DataFrame with 15 feature columns
        """
        df = df.copy()
        
        # Get volume column
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        
        # 1. Normalized close (z-score over rolling window)
        close = df['close'].astype(float)
        rolling_mean = close.rolling(window=50, min_periods=1).mean()
        rolling_std = close.rolling(window=50, min_periods=1).std()
        df['close'] = (close - rolling_mean) / (rolling_std + 1e-8)
        
        # 2. Simple returns
        df['returns'] = close.pct_change().fillna(0)
        
        # 3. Log returns
        df['log_returns'] = np.log(close / close.shift(1)).fillna(0)
        
        # 4. SMA Fast
        sma_fast = close.rolling(window=self.sma_fast_period, min_periods=1).mean()
        df['sma_fast'] = (sma_fast - rolling_mean) / (rolling_std + 1e-8)
        
        # 5. SMA Slow
        sma_slow = close.rolling(window=self.sma_slow_period, min_periods=1).mean()
        df['sma_slow'] = (sma_slow - rolling_mean) / (rolling_std + 1e-8)
        
        # 6. SMA Ratio (trend indicator)
        df['sma_ratio'] = sma_fast / (sma_slow + 1e-8) - 1.0  # Centered around 0
        
        # 7. ATR
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period, min_periods=1).mean()
        
        # Normalize ATR
        atr_rolling_std = atr.rolling(window=50, min_periods=1).std()
        df['atr'] = (atr - atr.rolling(window=50, min_periods=1).mean()) / (atr_rolling_std + 1e-8)
        
        # 8. ATR Normalized by close (volatility relative to price)
        df['atr_norm'] = atr / (close + 1e-8)
        
        # 9. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # 10. Bollinger Band Position
        bb_sma = close.rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=1).std()
        bb_upper = bb_sma + self.bb_std * bb_std
        bb_lower = bb_sma - self.bb_std * bb_std
        
        # Position in band: -1 (at lower), 0 (at middle), +1 (at upper)
        df['bb_position'] = 2 * (close - bb_lower) / (bb_upper - bb_lower + 1e-8) - 1
        
        # 11. Volume Ratio
        volume = df[vol_col].astype(float)
        volume_ma = volume.rolling(window=self.volume_ma_period, min_periods=1).mean()
        df['volume_ratio'] = (volume / (volume_ma + 1e-8)) - 1.0  # Centered around 0
        
        # 12-15. SMC Features
        df = self._add_smc_features(df)
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Clip extreme values
        for col in self.FEATURE_NAMES:
            if col in df.columns:
                df[col] = df[col].clip(-10, 10)
        
        return df[self.FEATURE_NAMES]
    
    def _add_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMC features: FVG and Swing Points."""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'] if 'close' in df.columns else df['close'].astype(float)
        open_price = df['open'].astype(float)
        
        # Fair Value Gaps (FVG)
        # Bullish FVG: gap between candle[i-2].high and candle[i].low
        df['SMC_FVG_Bullish'] = (
            (low > high.shift(2)) & 
            (close.shift(1) > open_price.shift(1))  # Middle candle is bullish
        ).astype(float)
        
        # Bearish FVG: gap between candle[i].high and candle[i-2].low
        df['SMC_FVG_Bearish'] = (
            (high < low.shift(2)) & 
            (close.shift(1) < open_price.shift(1))  # Middle candle is bearish
        ).astype(float)
        
        # Swing Points
        # Swing High: high is higher than surrounding bars
        rolling_high = high.rolling(window=self.swing_lookback * 2 + 1, center=True, min_periods=1).max()
        df['SMC_Swing_High'] = (high == rolling_high).astype(float)
        
        # Swing Low: low is lower than surrounding bars
        rolling_low = low.rolling(window=self.swing_lookback * 2 + 1, center=True, min_periods=1).min()
        df['SMC_Swing_Low'] = (low == rolling_low).astype(float)
        
        return df
    
    @staticmethod
    def get_feature_dim() -> int:
        """Return number of features."""
        return 15
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Return feature names."""
        return V3Features.FEATURE_NAMES.copy()


class V4LiteDataset:
    """
    Dataset for v4 Lite that uses v3-style engineered features.
    
    This creates sequences of engineered features for training.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        seq_len: int = 50,
        feature_generator: Optional[V3Features] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with OHLCV data
            labels: Optional Series with direction labels
            seq_len: Sequence length (default: 50 like v3)
            feature_generator: V3Features instance
        """
        self.seq_len = seq_len
        self.feature_generator = feature_generator or V3Features()
        
        # Extract features
        self.features = self.feature_generator.extract_features(df)
        self.data = self.features.values.astype(np.float32)
        
        # Store labels
        if labels is not None:
            self.labels = labels.values.astype(np.int64)
        else:
            self.labels = None
        
        # Valid indices (enough history)
        self.valid_indices = list(range(seq_len, len(self.data)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get a sample.
        
        Returns:
            x: (seq_len, n_features) sequence
            y: label (optional)
        """
        real_idx = self.valid_indices[idx]
        
        # Get sequence ending at real_idx
        x = self.data[real_idx - self.seq_len : real_idx]
        
        # Get label
        y = self.labels[real_idx] if self.labels is not None else None
        
        return x, y
    
    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return self.data.shape[1]


if __name__ == "__main__":
    print("=" * 60)
    print("V3 Feature Engineering Test")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2025-01-01', periods=n, freq='5min')
    base_price = 2600 + np.cumsum(np.random.randn(n) * 2)
    
    df = pd.DataFrame({
        'time': dates,
        'open': base_price + np.random.randn(n) * 1,
        'high': base_price + np.abs(np.random.randn(n) * 2),
        'low': base_price - np.abs(np.random.randn(n) * 2),
        'close': base_price + np.random.randn(n) * 1,
        'tick_volume': np.random.randint(100, 10000, n),
    })
    
    # Extract features
    feature_gen = V3Features()
    features = feature_gen.extract_features(df)
    
    print(f"\n📊 Input shape: {df.shape}")
    print(f"📊 Output shape: {features.shape}")
    print(f"📊 Feature names: {feature_gen.get_feature_names()}")
    
    print(f"\n📈 Feature statistics:")
    print(features.describe())
    
    # Check for NaN/Inf
    print(f"\n🔍 NaN values: {features.isna().sum().sum()}")
    print(f"🔍 Inf values: {np.isinf(features.values).sum()}")
    
    # Test dataset
    print("\n" + "=" * 60)
    print("V4 Lite Dataset Test")
    print("=" * 60)
    
    labels = pd.Series(np.random.randint(0, 3, n))
    dataset = V4LiteDataset(df, labels, seq_len=50)
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"📊 Sample shape: x={x.shape}, y={y}")
    
    # Check ranges
    print(f"\n📊 X range: [{x.min():.2f}, {x.max():.2f}]")
    
    print("\n✅ V3 Feature Engineering test passed!")

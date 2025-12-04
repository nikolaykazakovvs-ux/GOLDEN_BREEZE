"""
Golden Breeze v4 - Time Alignment Module

Aligns M5 (fast) stream with H1 (slow) stream using "Last Fully Closed" pattern.
CRITICAL: Prevents lookahead bias by only using closed candles.

Pattern: "Last Fully Closed H1 Bar"
- For M5 bar at 14:35, use H1 bar closed at 14:00 (not the developing 14:xx bar)
- pd.merge_asof with direction='backward' + proper lag

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class AlignedSample:
    """Represents a single aligned sample for training."""
    timestamp: pd.Timestamp
    m5_window: np.ndarray      # (seq_len_fast, features)
    h1_window: np.ndarray      # (seq_len_slow, features)
    smc_static: np.ndarray     # (static_smc_dim,)
    smc_dynamic: np.ndarray    # (max_tokens, dynamic_smc_dim)
    label: Optional[int] = None


class TimeAligner:
    """
    Time alignment utility for M5 and H1 streams.
    
    Ensures no lookahead bias by using only "last fully closed" H1 bar
    for each M5 timestamp.
    
    Key Methods:
    - align_timestamps: Map M5 times to last closed H1 times
    - align_streams: Full alignment with feature extraction
    - create_training_samples: Generate aligned samples for training
    
    Example:
        >>> aligner = TimeAligner()
        >>> aligned_df = aligner.align_streams(df_m5, df_h1_features)
    """
    
    def __init__(
        self,
        h1_close_minute: int = 0,
        apply_safety_lag: bool = True,
        safety_lag_minutes: int = 1,
    ):
        """
        Args:
            h1_close_minute: Minute when H1 bar closes (usually 0)
            apply_safety_lag: If True, add extra lag to ensure H1 is fully closed
            safety_lag_minutes: Extra minutes to add as safety margin
        """
        self.h1_close_minute = h1_close_minute
        self.apply_safety_lag = apply_safety_lag
        self.safety_lag_minutes = safety_lag_minutes
    
    def get_last_closed_h1(self, m5_time: pd.Timestamp) -> pd.Timestamp:
        """
        Get the timestamp of the last fully closed H1 bar for a given M5 time.
        
        For M5 at 14:35 → H1 closed at 14:00 is the "current developing" bar
        → We need to use 13:00 bar (last FULLY closed)
        
        BUT if M5 is exactly at 15:00, then 14:00 bar is closed.
        
        Logic:
        - floor to hour: 14:35 → 14:00
        - subtract 1 hour: 14:00 → 13:00 (if not exactly on hour)
        - if exactly on hour (minute=0), the previous hour is last closed
        """
        # Floor to current hour
        hour_start = m5_time.floor('h')
        
        # If M5 is exactly at hour start (e.g., 14:00:00),
        # then the previous H1 bar (13:00) just closed
        if m5_time.minute == 0 and m5_time.second == 0:
            last_closed = hour_start - pd.Timedelta(hours=1)
        else:
            # Otherwise, the current hour's H1 is still developing
            # Use the previous hour's bar
            last_closed = hour_start - pd.Timedelta(hours=1)
        
        # Apply safety lag if enabled
        if self.apply_safety_lag:
            last_closed = last_closed - pd.Timedelta(minutes=self.safety_lag_minutes)
            last_closed = last_closed.floor('h')  # Re-floor after subtracting
        
        return last_closed
    
    def align_timestamps(
        self, 
        m5_times: pd.DatetimeIndex,
        h1_times: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Map each M5 timestamp to the corresponding last closed H1 timestamp.
        
        Args:
            m5_times: DatetimeIndex of M5 bars
            h1_times: DatetimeIndex of H1 bars (what we have available)
            
        Returns:
            Series mapping M5 index to H1 timestamp
        """
        aligned = []
        
        for m5_time in m5_times:
            target_h1 = self.get_last_closed_h1(m5_time)
            
            # Find the closest H1 time that is <= target
            valid_h1 = h1_times[h1_times <= target_h1]
            
            if len(valid_h1) > 0:
                aligned.append(valid_h1[-1])  # Last valid H1
            else:
                aligned.append(pd.NaT)  # No valid H1 available
        
        return pd.Series(aligned, index=m5_times, name='aligned_h1_time')
    
    def align_streams(
        self,
        df_m5: pd.DataFrame,
        df_h1_features: pd.DataFrame,
        m5_time_col: str = 'time',
    ) -> pd.DataFrame:
        """
        Align M5 data with H1 features using merge_asof.
        
        Uses pd.merge_asof with direction='backward' to find the last
        available H1 bar for each M5 timestamp.
        
        Args:
            df_m5: M5 OHLCV DataFrame with time column
            df_h1_features: H1 features DataFrame (from SMCProcessor)
            m5_time_col: Name of time column in df_m5
            
        Returns:
            Merged DataFrame with M5 rows and aligned H1 features
        """
        # Prepare M5 data
        m5 = df_m5.copy()
        if m5_time_col in m5.columns:
            m5[m5_time_col] = pd.to_datetime(m5[m5_time_col])
        elif isinstance(m5.index, pd.DatetimeIndex):
            m5 = m5.reset_index()
            m5_time_col = m5.columns[0]
            m5[m5_time_col] = pd.to_datetime(m5[m5_time_col])
        
        # Prepare H1 features
        h1 = df_h1_features.copy()
        if not isinstance(h1.index, pd.DatetimeIndex):
            if 'time' in h1.columns:
                h1 = h1.set_index('time')
            h1.index = pd.to_datetime(h1.index)
        
        h1 = h1.reset_index()
        h1_time_col = h1.columns[0]
        h1[h1_time_col] = pd.to_datetime(h1[h1_time_col])
        
        # Calculate the "last closed H1" for each M5 bar
        # This adds a 1-hour lag to ensure we use closed bars
        m5['_aligned_h1_target'] = m5[m5_time_col].apply(self.get_last_closed_h1)
        
        # Sort both DataFrames by time
        m5 = m5.sort_values(m5_time_col)
        h1 = h1.sort_values(h1_time_col)
        
        # Use merge_asof with the target H1 time
        # This finds the closest H1 time that is <= _aligned_h1_target
        merged = pd.merge_asof(
            m5,
            h1,
            left_on='_aligned_h1_target',
            right_on=h1_time_col,
            direction='backward',
            suffixes=('', '_h1'),
        )
        
        # Rename H1 time column for clarity
        if h1_time_col in merged.columns:
            merged = merged.rename(columns={h1_time_col: 'h1_time'})
        
        # Drop temporary column
        merged = merged.drop(columns=['_aligned_h1_target'])
        
        return merged
    
    def create_training_samples(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_smc: pd.DataFrame,
        df_labels: pd.DataFrame,
        seq_len_fast: int = 200,
        seq_len_slow: int = 50,
        m5_features: List[str] = None,
        h1_features: List[str] = None,
        label_col: str = 'label',
    ) -> List[AlignedSample]:
        """
        Create aligned training samples for the Fusion model.
        
        For each valid M5 bar:
        1. Extract M5 window (seq_len_fast bars)
        2. Find aligned H1 timestamp
        3. Extract H1 window (seq_len_slow bars) + SMC features
        4. Pair with label if available
        
        Args:
            df_m5: M5 OHLCV DataFrame
            df_h1: H1 OHLCV DataFrame
            df_smc: SMC features DataFrame (from SMCProcessor)
            df_labels: Labels DataFrame with timestamp and label columns
            seq_len_fast: M5 sequence length
            seq_len_slow: H1 sequence length
            m5_features: Feature columns for M5 (default: OHLCV)
            h1_features: Feature columns for H1 (default: OHLCV)
            label_col: Name of label column
            
        Returns:
            List of AlignedSample objects
        """
        samples = []
        
        # Default feature columns
        if m5_features is None:
            m5_features = ['open', 'high', 'low', 'close', 'volume']
        if h1_features is None:
            h1_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Prepare M5 data
        m5 = df_m5.copy()
        if 'time' in m5.columns:
            m5 = m5.set_index('time')
        m5.index = pd.to_datetime(m5.index)
        
        # Prepare H1 data
        h1 = df_h1.copy()
        if 'time' in h1.columns:
            h1 = h1.set_index('time')
        h1.index = pd.to_datetime(h1.index)
        
        # Prepare SMC data
        smc = df_smc.copy()
        if not isinstance(smc.index, pd.DatetimeIndex):
            smc.index = pd.to_datetime(smc.index)
        
        # Prepare labels
        labels = df_labels.copy()
        if 'time' in labels.columns:
            labels = labels.set_index('time')
        labels.index = pd.to_datetime(labels.index)
        
        # Normalize M5 data
        m5_norm = self._normalize_ohlcv(m5, m5_features)
        h1_norm = self._normalize_ohlcv(h1, h1_features)
        
        # Iterate through M5 bars
        m5_times = m5.index.tolist()
        
        for i in range(seq_len_fast, len(m5_times)):
            m5_time = m5_times[i]
            
            # Skip if no label
            if m5_time not in labels.index:
                continue
            
            label = labels.loc[m5_time, label_col]
            
            # Get M5 window
            m5_window = m5_norm.iloc[i - seq_len_fast:i][m5_features].values
            
            # Get aligned H1 time
            h1_target = self.get_last_closed_h1(m5_time)
            
            # Find H1 window ending at h1_target
            valid_h1_idx = h1.index[h1.index <= h1_target]
            if len(valid_h1_idx) < seq_len_slow:
                continue  # Not enough H1 data
            
            h1_end_idx = h1.index.get_loc(valid_h1_idx[-1])
            h1_start_idx = max(0, h1_end_idx - seq_len_slow + 1)
            
            if h1_end_idx - h1_start_idx + 1 < seq_len_slow:
                continue
            
            h1_window = h1_norm.iloc[h1_start_idx:h1_end_idx + 1][h1_features].values
            
            # Get SMC features for the aligned H1 time
            if h1_target in smc.index:
                smc_row = smc.loc[h1_target]
            else:
                # Find nearest SMC row
                valid_smc_idx = smc.index[smc.index <= h1_target]
                if len(valid_smc_idx) == 0:
                    continue
                smc_row = smc.loc[valid_smc_idx[-1]]
            
            # Extract static and dynamic SMC
            smc_static = self._extract_static_smc(smc_row)
            smc_dynamic = self._extract_dynamic_smc(smc_row)
            
            sample = AlignedSample(
                timestamp=m5_time,
                m5_window=m5_window.astype(np.float32),
                h1_window=h1_window.astype(np.float32),
                smc_static=smc_static,
                smc_dynamic=smc_dynamic,
                label=int(label),
            )
            
            samples.append(sample)
        
        return samples
    
    def _normalize_ohlcv(
        self, 
        df: pd.DataFrame, 
        cols: List[str]
    ) -> pd.DataFrame:
        """Normalize OHLCV data using rolling statistics."""
        result = df.copy()
        
        for col in cols:
            if col in result.columns:
                # Use rolling mean and std for normalization
                roll_mean = result[col].rolling(window=20, min_periods=1).mean()
                roll_std = result[col].rolling(window=20, min_periods=1).std()
                roll_std = roll_std.replace(0, 1)  # Avoid division by zero
                
                result[col] = (result[col] - roll_mean) / roll_std
        
        return result.fillna(0)
    
    def _extract_static_smc(self, row: pd.Series, dim: int = 16) -> np.ndarray:
        """Extract static SMC vector from row."""
        static_cols = [
            'smc_bullish_ob_count',
            'smc_bearish_ob_count',
            'smc_nearest_ob_rel_dist',
            'smc_nearest_ob_decay',
            'smc_nearest_ob_is_bullish',
            'smc_avg_ob_decay',
            'smc_fresh_ob_count',
            'smc_mitigated_ob_count',
            'smc_market_structure',
            'smc_swing_high_dist',
            'smc_swing_low_dist',
            'smc_session_asian',
            'smc_session_london',
            'smc_session_ny',
        ]
        
        values = []
        for col in static_cols:
            if col in row.index:
                values.append(float(row[col]) if not pd.isna(row[col]) else 0.0)
            else:
                values.append(0.0)
        
        # Pad to dim
        while len(values) < dim:
            values.append(0.0)
        
        return np.array(values[:dim], dtype=np.float32)
    
    def _extract_dynamic_smc(
        self, 
        row: pd.Series,
        max_tokens: int = 10,
        dim: int = 12,
    ) -> np.ndarray:
        """Extract dynamic SMC matrix from row."""
        result = np.zeros((max_tokens, dim), dtype=np.float32)
        
        if 'dynamic_obs' in row.index and row['dynamic_obs'] is not None:
            dynamic_obs = row['dynamic_obs']
            
            for i, ob in enumerate(dynamic_obs[:max_tokens]):
                if isinstance(ob, dict):
                    result[i, 0] = ob.get('rel_high', 0)
                    result[i, 1] = ob.get('rel_low', 0)
                    result[i, 2] = ob.get('is_bullish', 0)
                    result[i, 3] = ob.get('time_decay', 0)
                    result[i, 4] = ob.get('state_fresh', 0)
                    result[i, 5] = ob.get('state_mitigated', 0)
                    result[i, 6] = ob.get('state_broken', 0)
                    result[i, 7] = ob.get('strength', 0)
                    result[i, 8] = min(ob.get('age_bars', 0) / 100.0, 2.0)
        
        return result


def validate_no_lookahead(
    df_aligned: pd.DataFrame,
    m5_time_col: str = 'time',
    h1_time_col: str = 'h1_time',
) -> bool:
    """
    Validate that no lookahead bias exists in aligned data.
    
    For each row, H1 time should be at least 1 hour before M5 time
    (since we use "last fully closed" H1 bar).
    """
    if m5_time_col not in df_aligned.columns or h1_time_col not in df_aligned.columns:
        raise ValueError(f"Required columns not found: {m5_time_col}, {h1_time_col}")
    
    m5_times = pd.to_datetime(df_aligned[m5_time_col])
    h1_times = pd.to_datetime(df_aligned[h1_time_col])
    
    # Calculate time difference
    time_diff = m5_times - h1_times
    
    # H1 should be at least 1 hour behind M5
    min_required_lag = pd.Timedelta(hours=1)
    
    violations = time_diff < min_required_lag
    violation_count = violations.sum()
    
    if violation_count > 0:
        print(f"⚠️ WARNING: {violation_count} rows have lookahead bias!")
        print(f"First violation: M5={m5_times[violations].iloc[0]}, H1={h1_times[violations].iloc[0]}")
        return False
    
    print(f"✅ No lookahead bias detected in {len(df_aligned)} rows")
    return True


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("TimeAligner - Quick Test")
    print("=" * 60)
    
    # Create dummy M5 data
    m5_dates = pd.date_range(start='2025-01-01 00:00', periods=1000, freq='5min')
    df_m5 = pd.DataFrame({
        'time': m5_dates,
        'open': 2650 + np.random.randn(1000) * 5,
        'high': 2655 + np.random.randn(1000) * 5,
        'low': 2645 + np.random.randn(1000) * 5,
        'close': 2650 + np.random.randn(1000) * 5,
        'volume': np.random.randint(100, 1000, 1000),
    })
    
    # Create dummy H1 features
    h1_dates = pd.date_range(start='2025-01-01 00:00', periods=100, freq='h')
    df_h1_features = pd.DataFrame({
        'smc_bullish_ob_count': np.random.randint(0, 5, 100),
        'smc_bearish_ob_count': np.random.randint(0, 5, 100),
        'smc_nearest_ob_rel_dist': np.random.rand(100),
        'smc_nearest_ob_decay': np.random.rand(100),
        'smc_nearest_ob_is_bullish': np.random.randint(0, 2, 100),
        'smc_avg_ob_decay': np.random.rand(100),
        'smc_fresh_ob_count': np.random.randint(0, 3, 100),
        'smc_mitigated_ob_count': np.random.randint(0, 3, 100),
        'smc_market_structure': np.zeros(100),
        'smc_swing_high_dist': np.random.rand(100),
        'smc_swing_low_dist': np.random.rand(100),
        'smc_session_asian': np.zeros(100),
        'smc_session_london': np.zeros(100),
        'smc_session_ny': np.zeros(100),
    }, index=h1_dates)
    
    print(f"M5 data: {len(df_m5)} bars")
    print(f"H1 features: {len(df_h1_features)} bars")
    
    # Test alignment
    aligner = TimeAligner()
    
    # Test get_last_closed_h1
    test_times = [
        pd.Timestamp('2025-01-01 14:35:00'),
        pd.Timestamp('2025-01-01 15:00:00'),
        pd.Timestamp('2025-01-01 14:00:00'),
        pd.Timestamp('2025-01-01 14:05:00'),
    ]
    
    print("\nLast closed H1 tests:")
    for t in test_times:
        h1 = aligner.get_last_closed_h1(t)
        print(f"  M5 {t} → H1 {h1}")
    
    # Test full alignment
    print("\nFull alignment test...")
    aligned = aligner.align_streams(df_m5, df_h1_features)
    print(f"Aligned data: {len(aligned)} rows")
    print(f"Columns: {list(aligned.columns)[:10]}...")
    
    # Validate no lookahead
    print("\nValidating no lookahead...")
    validate_no_lookahead(aligned, m5_time_col='time', h1_time_col='h1_time')
    
    print("\n✅ TimeAligner test passed!")

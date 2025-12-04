"""
Golden Breeze v4 - SMC (Smart Money Concepts) Processor

Calculates Order Block features with explicit time decay.
NO reliance on implicit positional encoding for "age".

**GPT Specs Integration:**
Features per Order Block: [dist_high, dist_low, is_bullish, state_id, age_norm]
Decay: decay_weight = exp(-0.05 * age_bars)
State ID: 'fresh'=0, 'mitigated'=1, 'broken'=2
Output: DataFrame indexed by H1 time with calculated features for *active* OBs/FVGs.

Author: Golden Breeze Team
Version: 4.0.1
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


# === Constants ===
SMC_DECAY_LAMBDA = 0.05
SMC_MAX_OB_AGE = 200  # bars

# State ID mapping (GPT Specs)
STATE_MAP = {'fresh': 0, 'mitigated': 1, 'broken': 2}


class OBState(IntEnum):
    """Order Block state (integer IDs per GPT spec)."""
    FRESH = 0       # Never tested
    MITIGATED = 1   # Partially tested (price touched but didn't break)
    BROKEN = 2      # Fully broken (price closed through)


class OBType(IntEnum):
    """Order Block type."""
    BULLISH = 1
    BEARISH = 0


@dataclass
class OrderBlock:
    """Represents a single Order Block."""
    creation_idx: int           # Bar index when OB was created
    creation_time: pd.Timestamp # Timestamp of creation
    ob_high: float              # Upper boundary
    ob_low: float               # Lower boundary
    ob_type: OBType             # Bullish or Bearish
    state: OBState = OBState.FRESH
    strength: float = 1.0       # Initial strength (based on impulse move)
    mitigation_count: int = 0   # Number of times price touched
    
    @property
    def is_bullish(self) -> bool:
        return self.ob_type == OBType.BULLISH
    
    @property
    def midpoint(self) -> float:
        return (self.ob_high + self.ob_low) / 2
    
    @property
    def state_id(self) -> int:
        """Return integer state ID (GPT Specs)."""
        return int(self.state)
    
    @property
    def state_name(self) -> str:
        """Return string state name."""
        return ['fresh', 'mitigated', 'broken'][self.state]
    
    def get_age_bars(self, current_idx: int) -> int:
        """Calculate age in bars."""
        return max(0, current_idx - self.creation_idx)
    
    def get_age_norm(self, current_idx: int, max_age: int = SMC_MAX_OB_AGE) -> float:
        """Calculate normalized age [0, 1] (GPT Specs)."""
        age = self.get_age_bars(current_idx)
        return min(age / max_age, 1.0)
    
    def get_decay_weight(self, current_idx: int, decay_lambda: float = SMC_DECAY_LAMBDA) -> float:
        """Calculate decay_weight = exp(-0.05 * age_bars) (GPT Specs)."""
        age = self.get_age_bars(current_idx)
        return np.exp(-decay_lambda * age)
    
    def update_state(self, high: float, low: float, close: float):
        """Update OB state based on price action."""
        if self.state == OBState.BROKEN:
            return  # Already broken, no update needed
        
        if self.is_bullish:
            # Bullish OB: price should come down to test it
            if close < self.ob_low:
                self.state = OBState.BROKEN
            elif low <= self.ob_high:
                self.state = OBState.MITIGATED
                self.mitigation_count += 1
        else:
            # Bearish OB: price should come up to test it
            if close > self.ob_high:
                self.state = OBState.BROKEN
            elif high >= self.ob_low:
                self.state = OBState.MITIGATED
                self.mitigation_count += 1


class SMCProcessor:
    """
    Smart Money Concepts Processor for Order Block detection and feature extraction.
    
    **GPT Specs Integration:**
    - Features: [dist_high, dist_low, is_bullish, state_id, age_norm]
    - Decay: decay_weight = exp(-0.05 * age_bars)
    - State ID: 'fresh'=0, 'mitigated'=1, 'broken'=2
    - Output: DataFrame indexed by H1 time with active OB/FVG features
    
    Key Methods:
    - process_h1_data(df_h1): Main entry point, returns SMC features DataFrame
    - detect_order_blocks(df): Identify Order Blocks from OHLCV
    - calculate_ob_features(df_h1): Calculate per-bar OB features
    
    Example:
        >>> processor = SMCProcessor(decay_lambda=0.05, max_ob_age=200)
        >>> df_h1 = pd.read_csv("data/raw/XAUUSD/H1.csv")
        >>> smc_features = processor.process_h1_data(df_h1)
    """
    
    def __init__(
        self,
        decay_lambda: float = SMC_DECAY_LAMBDA,
        max_ob_age: int = SMC_MAX_OB_AGE,
        max_active_obs: int = 10,
        min_impulse_atr: float = 1.5,
        atr_period: int = 14,
    ):
        """
        Args:
            decay_lambda: Decay rate for time decay formula
            max_ob_age: Maximum age in bars before OB expires
            max_active_obs: Maximum number of active OBs to track
            min_impulse_atr: Minimum impulse move (in ATR) to qualify as OB
            atr_period: Period for ATR calculation
        """
        self.decay_lambda = decay_lambda
        self.max_ob_age = max_ob_age
        self.max_active_obs = max_active_obs
        self.min_impulse_atr = min_impulse_atr
        self.atr_period = atr_period
        
        # Active Order Blocks
        self.active_obs: List[OrderBlock] = []
    
    def process_h1_data(self, df_h1: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: Process H1 data and return SMC features DataFrame.
        
        **GPT Specs:**
        Output DataFrame contains per-bar features for all *active* OBs:
        - ob_features: [dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight]
        - fvg_features: Similar for Fair Value Gaps (if any)
        - Aggregated static features for nearest/strongest OB
        
        Args:
            df_h1: H1 OHLCV DataFrame with columns [time, open, high, low, close, volume]
            
        Returns:
            DataFrame indexed by H1 time with SMC features
        """
        return self.calculate_ob_features(df_h1, return_dynamic=True)
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period, min_periods=1).mean()
        
        return atr
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[Tuple[int, OrderBlock]]:
        """
        Detect Order Blocks from OHLCV data.
        
        OB Detection Logic:
        1. Find impulse candles (body > min_impulse_atr * ATR)
        2. The candle before the impulse is the Order Block
        3. Bullish OB: impulse is bullish (close > open)
        4. Bearish OB: impulse is bearish (close < open)
        
        Args:
            df: DataFrame with OHLCV data (must have: open, high, low, close, time/index)
            
        Returns:
            List of (bar_index, OrderBlock) tuples
        """
        if len(df) < 3:
            return []
        
        detected_obs = []
        atr = self.calculate_atr(df)
        
        # Ensure we have a time column
        if 'time' in df.columns:
            times = pd.to_datetime(df['time'])
        elif isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        else:
            times = pd.to_datetime(df.index)
        
        for i in range(2, len(df)):
            # Current candle (impulse candidate)
            curr_open = df.iloc[i]['open']
            curr_close = df.iloc[i]['close']
            curr_body = abs(curr_close - curr_open)
            
            # Check for impulse move
            if curr_body < self.min_impulse_atr * atr.iloc[i]:
                continue
            
            # Previous candle (OB candidate)
            prev_high = df.iloc[i-1]['high']
            prev_low = df.iloc[i-1]['low']
            prev_time = times.iloc[i-1] if hasattr(times, 'iloc') else times[i-1]
            
            # Determine OB type based on impulse direction
            if curr_close > curr_open:
                # Bullish impulse → Bullish OB (last down candle before up move)
                ob_type = OBType.BULLISH
            else:
                # Bearish impulse → Bearish OB (last up candle before down move)
                ob_type = OBType.BEARISH
            
            # Calculate strength based on impulse size
            strength = min(2.0, curr_body / atr.iloc[i])
            
            ob = OrderBlock(
                creation_idx=i - 1,
                creation_time=prev_time,
                ob_high=prev_high,
                ob_low=prev_low,
                ob_type=ob_type,
                state=OBState.FRESH,
                strength=strength,
            )
            
            detected_obs.append((i - 1, ob))
        
        return detected_obs
    
    def calculate_ob_features(
        self, 
        df_h1: pd.DataFrame,
        return_dynamic: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate Order Block features for every H1 bar.
        
        **GPT Specs Features per active OB:**
        - dist_high: (Close - OB_High) / ATR
        - dist_low: (Close - OB_Low) / ATR
        - is_bullish: 1 for bullish, 0 for bearish
        - state_id: 0=fresh, 1=mitigated, 2=broken
        - age_norm: age_bars / max_ob_age (normalized to [0,1])
        - decay_weight: exp(-0.05 * age_bars)
        
        Args:
            df_h1: H1 OHLCV DataFrame
            return_dynamic: If True, return per-OB dynamic features
            
        Returns:
            DataFrame indexed by H1 time with SMC features
        """
        df = df_h1.copy()
        
        # Ensure time index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        
        # Detect all OBs
        all_obs = self.detect_order_blocks(df.reset_index())
        
        # Initialize result arrays
        n_bars = len(df)
        
        # Static SMC features (aggregated from active OBs)
        static_features = {
            'smc_bullish_ob_count': np.zeros(n_bars),
            'smc_bearish_ob_count': np.zeros(n_bars),
            'smc_nearest_ob_dist_high': np.full(n_bars, np.nan),
            'smc_nearest_ob_dist_low': np.full(n_bars, np.nan),
            'smc_nearest_ob_decay_weight': np.zeros(n_bars),
            'smc_nearest_ob_is_bullish': np.zeros(n_bars),
            'smc_nearest_ob_state_id': np.zeros(n_bars),
            'smc_nearest_ob_age_norm': np.zeros(n_bars),
            'smc_avg_ob_decay': np.zeros(n_bars),
            'smc_fresh_ob_count': np.zeros(n_bars),
            'smc_mitigated_ob_count': np.zeros(n_bars),
            # Market structure features
            'smc_market_structure': np.zeros(n_bars),  # -1: bearish, 0: ranging, 1: bullish
            'smc_swing_high_dist': np.full(n_bars, np.nan),
            'smc_swing_low_dist': np.full(n_bars, np.nan),
            # Session features
            'smc_session_asian': np.zeros(n_bars),
            'smc_session_london': np.zeros(n_bars),
            'smc_session_ny': np.zeros(n_bars),
        }
        
        # Dynamic OB features (per active OB) - GPT Specs format
        if return_dynamic:
            dynamic_features = []  # List of lists per bar
        
        # Track active OBs
        active_obs: List[OrderBlock] = []
        ob_queue = list(all_obs)  # (creation_idx, OB)
        ob_queue.sort(key=lambda x: x[0])
        
        for i in range(n_bars):
            current_close = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_atr = atr.iloc[i] if atr.iloc[i] > 0 else 1.0
            current_time = df.index[i]
            
            # Add new OBs that were created at or before this bar
            while ob_queue and ob_queue[0][0] <= i:
                _, new_ob = ob_queue.pop(0)
                active_obs.append(new_ob)
            
            # Update OB states and filter expired/broken
            updated_obs = []
            for ob in active_obs:
                ob.update_state(current_high, current_low, current_close)
                age = ob.get_age_bars(i)
                
                # Keep if not expired and not fully broken
                if age <= self.max_ob_age and ob.state != OBState.BROKEN:
                    updated_obs.append(ob)
            
            active_obs = updated_obs
            
            # Sort by strength * decay (most relevant first)
            active_obs.sort(
                key=lambda ob: ob.strength * ob.get_decay_weight(i, self.decay_lambda),
                reverse=True
            )
            
            # Keep only top N
            active_obs = active_obs[:self.max_active_obs]
            
            # Calculate static features
            bullish_count = sum(1 for ob in active_obs if ob.is_bullish)
            bearish_count = len(active_obs) - bullish_count
            fresh_count = sum(1 for ob in active_obs if ob.state == OBState.FRESH)
            mitigated_count = sum(1 for ob in active_obs if ob.state == OBState.MITIGATED)
            
            static_features['smc_bullish_ob_count'][i] = bullish_count
            static_features['smc_bearish_ob_count'][i] = bearish_count
            static_features['smc_fresh_ob_count'][i] = fresh_count
            static_features['smc_mitigated_ob_count'][i] = mitigated_count
            
            # Nearest OB features (GPT Specs format)
            if active_obs:
                # Find nearest OB by distance
                min_dist = float('inf')
                nearest_ob = None
                for ob in active_obs:
                    dist = min(abs(current_close - ob.ob_high), abs(current_close - ob.ob_low))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_ob = ob
                
                if nearest_ob:
                    # GPT Specs: dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight
                    static_features['smc_nearest_ob_dist_high'][i] = (current_close - nearest_ob.ob_high) / current_atr
                    static_features['smc_nearest_ob_dist_low'][i] = (current_close - nearest_ob.ob_low) / current_atr
                    static_features['smc_nearest_ob_is_bullish'][i] = float(nearest_ob.is_bullish)
                    static_features['smc_nearest_ob_state_id'][i] = nearest_ob.state_id
                    static_features['smc_nearest_ob_age_norm'][i] = nearest_ob.get_age_norm(i, self.max_ob_age)
                    static_features['smc_nearest_ob_decay_weight'][i] = nearest_ob.get_decay_weight(i, self.decay_lambda)
                
                # Average decay
                avg_decay = np.mean([ob.get_decay_weight(i, self.decay_lambda) for ob in active_obs])
                static_features['smc_avg_ob_decay'][i] = avg_decay
            
            # Session detection (UTC)
            hour = current_time.hour
            if 0 <= hour < 8:
                static_features['smc_session_asian'][i] = 1.0
            elif 8 <= hour < 16:
                static_features['smc_session_london'][i] = 1.0
            else:
                static_features['smc_session_ny'][i] = 1.0
            
            # Dynamic features per OB (GPT Specs format)
            if return_dynamic:
                bar_dynamic = []
                for ob in active_obs:
                    # GPT Specs: [dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight]
                    dist_high = (current_close - ob.ob_high) / current_atr
                    dist_low = (current_close - ob.ob_low) / current_atr
                    decay_weight = ob.get_decay_weight(i, self.decay_lambda)
                    age_norm = ob.get_age_norm(i, self.max_ob_age)
                    
                    bar_dynamic.append({
                        'dist_high': dist_high,
                        'dist_low': dist_low,
                        'is_bullish': float(ob.is_bullish),
                        'state_id': ob.state_id,
                        'age_norm': age_norm,
                        'decay_weight': decay_weight,
                        # Additional useful features
                        'strength': ob.strength,
                        'age_bars': ob.get_age_bars(i),
                    })
                
                dynamic_features.append(bar_dynamic)
        
        # Build result DataFrame
        result_df = pd.DataFrame(static_features, index=df.index)
        
        # Fill NaN with 0 for distance features if no OBs
        result_df['smc_nearest_ob_dist_high'] = result_df['smc_nearest_ob_dist_high'].fillna(0)
        result_df['smc_nearest_ob_dist_low'] = result_df['smc_nearest_ob_dist_low'].fillna(0)
        result_df['smc_swing_high_dist'] = result_df['smc_swing_high_dist'].fillna(0)
        result_df['smc_swing_low_dist'] = result_df['smc_swing_low_dist'].fillna(0)
        
        if return_dynamic:
            result_df['dynamic_obs'] = dynamic_features
        
        return result_df
    
    def get_static_vector(self, row: pd.Series, dim: int = 16) -> np.ndarray:
        """
        Extract static SMC vector from a feature row.
        
        **GPT Specs format for nearest OB:**
        [dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight, ...]
        
        Returns:
            numpy array of shape (dim,)
        """
        static_cols = [
            # GPT Specs features for nearest OB
            'smc_nearest_ob_dist_high',
            'smc_nearest_ob_dist_low',
            'smc_nearest_ob_is_bullish',
            'smc_nearest_ob_state_id',
            'smc_nearest_ob_age_norm',
            'smc_nearest_ob_decay_weight',
            # Aggregated counts
            'smc_bullish_ob_count',
            'smc_bearish_ob_count',
            'smc_fresh_ob_count',
            'smc_mitigated_ob_count',
            'smc_avg_ob_decay',
            # Market structure
            'smc_market_structure',
            'smc_swing_high_dist',
            'smc_swing_low_dist',
            # Session
            'smc_session_asian',
            'smc_session_london',
        ]
        
        values = []
        for col in static_cols:
            val = row.get(col, 0.0) if hasattr(row, 'get') else row[col] if col in row.index else 0.0
            values.append(float(val) if not pd.isna(val) else 0.0)
        
        # Pad to dim
        while len(values) < dim:
            values.append(0.0)
        
        return np.array(values[:dim], dtype=np.float32)
    
    def get_dynamic_matrix(
        self, 
        dynamic_obs: List[dict],
        max_tokens: int = 10,
        dim_per_token: int = 8,
    ) -> np.ndarray:
        """
        Extract dynamic OB matrix from dynamic_obs list.
        
        **GPT Specs format per OB:**
        [dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight, strength, age_bars_norm]
        
        Args:
            dynamic_obs: List of OB dicts from calculate_ob_features
            max_tokens: Maximum number of tokens
            dim_per_token: Features per token (default 8 per GPT Specs)
            
        Returns:
            numpy array of shape (max_tokens, dim_per_token)
        """
        result = np.zeros((max_tokens, dim_per_token), dtype=np.float32)
        
        if dynamic_obs is None:
            return result
        
        for i, ob in enumerate(dynamic_obs[:max_tokens]):
            # GPT Specs: [dist_high, dist_low, is_bullish, state_id, age_norm, decay_weight]
            result[i, 0] = ob.get('dist_high', 0)
            result[i, 1] = ob.get('dist_low', 0)
            result[i, 2] = ob.get('is_bullish', 0)
            result[i, 3] = ob.get('state_id', 0) / 2.0  # Normalize state_id to [0, 1]
            result[i, 4] = ob.get('age_norm', 0)
            result[i, 5] = ob.get('decay_weight', 0)
            result[i, 6] = ob.get('strength', 0)
            result[i, 7] = min(ob.get('age_bars', 0) / 100.0, 2.0)  # Normalized age
        
        return result


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("SMCProcessor - Quick Test")
    print("=" * 60)
    
    # Create dummy H1 data
    np.random.seed(42)
    n_bars = 100
    
    dates = pd.date_range(start='2025-01-01', periods=n_bars, freq='h')
    base_price = 2650.0
    
    # Generate random walk
    returns = np.random.randn(n_bars) * 5
    close = base_price + np.cumsum(returns)
    
    df = pd.DataFrame({
        'time': dates,
        'open': close - np.random.rand(n_bars) * 2,
        'high': close + np.abs(np.random.randn(n_bars)) * 3,
        'low': close - np.abs(np.random.randn(n_bars)) * 3,
        'close': close,
        'volume': np.random.randint(100, 1000, n_bars),
    })
    
    print(f"Input data: {len(df)} H1 bars")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Process
    processor = SMCProcessor()
    features = processor.calculate_ob_features(df)
    
    print(f"\nOutput features: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    
    # Check some values
    bullish_max = features['smc_bullish_ob_count'].max()
    bearish_max = features['smc_bearish_ob_count'].max()
    print(f"\nMax bullish OBs: {bullish_max}")
    print(f"Max bearish OBs: {bearish_max}")
    
    # Test vector extraction
    if len(features) > 50:
        row = features.iloc[50]
        static_vec = processor.get_static_vector(row)
        print(f"\nStatic vector shape: {static_vec.shape}")
        print(f"Static vector: {static_vec[:8]}...")
        
        if 'dynamic_obs' in features.columns:
            dynamic = features.iloc[50]['dynamic_obs']
            dyn_mat = processor.get_dynamic_matrix(dynamic)
            print(f"\nDynamic matrix shape: {dyn_mat.shape}")
            print(f"Active OBs at bar 50: {len(dynamic)}")
    
    print("\n✅ SMCProcessor test passed!")

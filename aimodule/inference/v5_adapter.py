"""
Golden Breeze v5 Ultimate - Inference Adapter

This module provides a production-ready inference pipeline for the v5 Ultimate model.
It handles:
1. Loading the trained model and scaler parameters
2. Preprocessing live MT5 data (indicators, normalization)
3. Running inference with proper tensor formatting
4. Converting logits to trading signals

Author: Golden Breeze Team
Version: 5.0.0 Ultimate
Date: 2025-12-05
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass

# Technical indicators
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("⚠️ ta library not found. Install with: pip install ta")


@dataclass
class PredictionResult:
    """Result of model prediction."""
    signal: str           # "UP", "DOWN", "NEUTRAL"
    confidence: float     # Probability of the signal (0-1)
    probabilities: Dict[str, float]  # All class probabilities
    raw_logits: Optional[np.ndarray] = None


class GoldenBreezeAdapter:
    """
    Inference adapter for Golden Breeze v5 Ultimate model.
    
    This class handles:
    - Model loading and initialization
    - Live data preprocessing (matching training pipeline)
    - Normalization using training statistics
    - Inference and signal generation
    
    Usage:
        adapter = GoldenBreezeAdapter()
        result = adapter.predict(df_m5, df_h1, strategy_features)
    """
    
    # Feature names for M5 (fast) timeframe - 15 features
    FAST_FEATURES = [
        'returns', 'log_returns', 'volatility', 'rsi_norm',
        'sma_slope_norm', 'price_position', 'atr_norm',
        'high_low_range', 'close_open_range', 'volume_norm',
        'momentum', 'roc', 'ema_cross', 'bb_position', 'trend_strength'
    ]
    
    # Feature names for H1 (slow) timeframe - 8 features
    SLOW_FEATURES = [
        'trend_direction', 'volatility_regime', 'momentum_h1',
        'support_distance', 'resistance_distance', 
        'session_progress', 'volume_profile', 'price_level'
    ]
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(
        self,
        model_path: str = "models/v5_ultimate/best_model.pt",
        scaler_path: str = "models/v5_ultimate/scaler_params.json",
        device: str = "auto",
        threshold: float = 0.4,  # Confidence threshold for signals
    ):
        """
        Initialize the adapter.
        
        Args:
            model_path: Path to trained model checkpoint
            scaler_path: Path to scaler parameters JSON
            device: 'cuda', 'cpu', or 'auto'
            threshold: Confidence threshold for UP/DOWN signals
        """
        self.threshold = threshold
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Initializing GoldenBreezeAdapter on {self.device}")
        
        # Load scaler parameters
        self._load_scalers(scaler_path)
        
        # Load model
        self._load_model(model_path)
        
        print("✅ GoldenBreezeAdapter ready!")
    
    def _load_scalers(self, scaler_path: str):
        """Load normalization parameters from JSON."""
        print(f"📊 Loading scaler parameters: {scaler_path}")
        
        with open(scaler_path, 'r') as f:
            params = json.load(f)
        
        # Convert to numpy arrays
        self.fast_mean = np.array(params['fast_mean'], dtype=np.float32)
        self.fast_std = np.array(params['fast_std'], dtype=np.float32)
        self.slow_mean = np.array(params['slow_mean'], dtype=np.float32)
        self.slow_std = np.array(params['slow_std'], dtype=np.float32)
        self.strat_mean = np.array(params['strat_mean'], dtype=np.float32)
        self.strat_std = np.array(params['strat_std'], dtype=np.float32)
        
        # Store metadata
        self.fast_seq_len = params.get('fast_seq_len', 50)
        self.slow_seq_len = params.get('slow_seq_len', 20)
        self.strat_features = params.get('strat_features', 64)
        
        print(f"   Loaded: fast({len(self.fast_mean)}), slow({len(self.slow_mean)}), strat({len(self.strat_mean)})")
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        print(f"🧠 Loading model: {model_path}")
        
        # Import model architecture
        from ..models.v5_ultimate import GoldenBreezeV5Ultimate, V5UltimateConfig
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with config from checkpoint if available
        if 'config' in checkpoint:
            config = V5UltimateConfig(**checkpoint['config'])
        else:
            config = V5UltimateConfig()
        
        self.model = GoldenBreezeV5Ultimate(config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get model info
        if 'val_mcc' in checkpoint:
            print(f"   Loaded epoch {checkpoint.get('epoch', '?')}, Val MCC: {checkpoint['val_mcc']:+.4f}")
        
        print(f"   Parameters: {self.model.count_parameters():,}")
    
    def _compute_m5_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute 15 features for M5 timeframe.
        
        Args:
            df: DataFrame with OHLCV data (at least 50 rows)
            
        Returns:
            Array of shape (seq_len, 15)
        """
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['volatility'] = df['returns'].rolling(14, min_periods=1).std().fillna(0)
        
        # RSI normalized to [-1, 1]
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().fillna(50)
            df['rsi_norm'] = (rsi - 50) / 50
        else:
            df['rsi_norm'] = 0.0
        
        # SMA slope normalized
        sma_fast = df['close'].rolling(10, min_periods=1).mean()
        df['sma_slope_norm'] = (sma_fast.diff() / df['close']).fillna(0)
        
        # Price position in range [0, 1]
        price_range = df['high'] - df['low']
        df['price_position'] = ((df['close'] - df['low']) / (price_range + 1e-8)).clip(0, 1)
        
        # ATR normalized by close
        if HAS_TA:
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['atr_norm'] = (atr / df['close']).fillna(0)
        else:
            df['atr_norm'] = price_range / df['close']
        
        # Range features
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['close']
        
        # Volume normalized (relative to mean)
        vol_mean = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_norm'] = (df['volume'] / (vol_mean + 1e-8) - 1).clip(-3, 3)
        
        # Momentum features
        df['momentum'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
        df['roc'] = df['close'].pct_change(5).fillna(0)
        
        # EMA cross signal
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_cross'] = ((ema_fast - ema_slow) / df['close']).fillna(0)
        
        # Bollinger position
        if HAS_TA:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_width = bb.bollinger_hband() - bb.bollinger_lband()
            df['bb_position'] = ((df['close'] - bb.bollinger_lband()) / (bb_width + 1e-8)).clip(0, 1).fillna(0.5)
        else:
            df['bb_position'] = 0.5
        
        # Trend strength (ADX-like)
        if HAS_TA:
            try:
                adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
                df['trend_strength'] = (adx / 100).fillna(0)
            except:
                df['trend_strength'] = 0.0
        else:
            df['trend_strength'] = 0.0
        
        # Extract features
        feature_cols = [
            'returns', 'log_returns', 'volatility', 'rsi_norm',
            'sma_slope_norm', 'price_position', 'atr_norm',
            'high_low_range', 'close_open_range', 'volume_norm',
            'momentum', 'roc', 'ema_cross', 'bb_position', 'trend_strength'
        ]
        
        features = df[feature_cols].values.astype(np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_h1_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute 8 features for H1 timeframe.
        
        Args:
            df: DataFrame with OHLCV data (at least 20 rows)
            
        Returns:
            Array of shape (seq_len, 8)
        """
        df = df.copy()
        
        # Trend direction (-1 to 1)
        sma_20 = df['close'].rolling(20, min_periods=1).mean()
        sma_50 = df['close'].rolling(50, min_periods=1).mean()
        df['trend_direction'] = ((sma_20 - sma_50) / (sma_50 + 1e-8)).clip(-1, 1).fillna(0)
        
        # Volatility regime (0 to 1)
        vol = df['close'].pct_change().rolling(14, min_periods=1).std()
        vol_sma = vol.rolling(50, min_periods=1).mean()
        df['volatility_regime'] = (vol / (vol_sma + 1e-8)).clip(0, 3).fillna(1) / 3
        
        # H1 momentum
        df['momentum_h1'] = (df['close'] / df['close'].shift(20) - 1).fillna(0)
        
        # Support/Resistance distances
        rolling_low = df['low'].rolling(50, min_periods=1).min()
        rolling_high = df['high'].rolling(50, min_periods=1).max()
        price_range = rolling_high - rolling_low
        
        df['support_distance'] = ((df['close'] - rolling_low) / (price_range + 1e-8)).clip(0, 1).fillna(0.5)
        df['resistance_distance'] = ((rolling_high - df['close']) / (price_range + 1e-8)).clip(0, 1).fillna(0.5)
        
        # Session progress (if timestamp available, else use position in data)
        if 'timestamp' in df.columns:
            try:
                ts = pd.to_datetime(df['timestamp'])
                df['session_progress'] = (ts.dt.hour * 60 + ts.dt.minute) / (24 * 60)
            except:
                df['session_progress'] = np.linspace(0, 1, len(df))
        else:
            df['session_progress'] = np.linspace(0, 1, len(df))
        
        # Volume profile
        vol_mean = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_profile'] = (df['volume'] / (vol_mean + 1e-8)).clip(0, 3).fillna(1) / 3
        
        # Price level (normalized 0-1 in recent range)
        df['price_level'] = df['support_distance']  # Reuse
        
        # Extract features
        feature_cols = [
            'trend_direction', 'volatility_regime', 'momentum_h1',
            'support_distance', 'resistance_distance',
            'session_progress', 'volume_profile', 'price_level'
        ]
        
        features = df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def preprocess_live_data(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        strategy_features: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess live MT5 data for model inference.
        
        Args:
            df_m5: M5 OHLCV DataFrame (at least 50 rows)
            df_h1: H1 OHLCV DataFrame (at least 20 rows)
            strategy_features: Optional (64,) array of strategy features.
                              If None, zeros will be used.
        
        Returns:
            Tuple of tensors: (x_fast, x_slow, x_strategy)
        """
        # Compute M5 features
        m5_features = self._compute_m5_features(df_m5)
        
        # Take last N rows
        if len(m5_features) > self.fast_seq_len:
            m5_features = m5_features[-self.fast_seq_len:]
        elif len(m5_features) < self.fast_seq_len:
            # Pad with zeros at the beginning
            pad = np.zeros((self.fast_seq_len - len(m5_features), m5_features.shape[1]))
            m5_features = np.vstack([pad, m5_features])
        
        # Compute H1 features
        h1_features = self._compute_h1_features(df_h1)
        
        if len(h1_features) > self.slow_seq_len:
            h1_features = h1_features[-self.slow_seq_len:]
        elif len(h1_features) < self.slow_seq_len:
            pad = np.zeros((self.slow_seq_len - len(h1_features), h1_features.shape[1]))
            h1_features = np.vstack([pad, h1_features])
        
        # Strategy features
        if strategy_features is None:
            strat_features = np.zeros(self.strat_features, dtype=np.float32)
        else:
            strat_features = np.array(strategy_features, dtype=np.float32)
            if len(strat_features) != self.strat_features:
                # Pad or truncate
                if len(strat_features) < self.strat_features:
                    strat_features = np.pad(strat_features, (0, self.strat_features - len(strat_features)))
                else:
                    strat_features = strat_features[:self.strat_features]
        
        # Normalize using training statistics
        m5_normalized = (m5_features - self.fast_mean) / (self.fast_std + 1e-6)
        h1_normalized = (h1_features - self.slow_mean) / (self.slow_std + 1e-6)
        strat_normalized = (strat_features - self.strat_mean) / (self.strat_std + 1e-6)
        
        # Convert to tensors with batch dimension
        x_fast = torch.from_numpy(m5_normalized).float().unsqueeze(0).to(self.device)   # (1, 50, 15)
        x_slow = torch.from_numpy(h1_normalized).float().unsqueeze(0).to(self.device)   # (1, 20, 8)
        x_strat = torch.from_numpy(strat_normalized).float().unsqueeze(0).to(self.device)  # (1, 64)
        
        return x_fast, x_slow, x_strat
    
    @torch.no_grad()
    def predict(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        strategy_features: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """
        Run inference on live data.
        
        Args:
            df_m5: M5 OHLCV DataFrame
            df_h1: H1 OHLCV DataFrame  
            strategy_features: Optional strategy feature vector
            
        Returns:
            PredictionResult with signal, confidence, and probabilities
        """
        # Preprocess
        x_fast, x_slow, x_strat = self.preprocess_live_data(df_m5, df_h1, strategy_features)
        
        # Run model
        logits = self.model(x_fast, x_slow, x_strat)  # (1, 3)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (3,)
        
        # Determine signal
        prob_down, prob_neutral, prob_up = probs
        
        if prob_up > self.threshold and prob_up > prob_down:
            signal = "UP"
            confidence = float(prob_up)
        elif prob_down > self.threshold and prob_down > prob_up:
            signal = "DOWN"
            confidence = float(prob_down)
        else:
            signal = "NEUTRAL"
            confidence = float(prob_neutral)
        
        return PredictionResult(
            signal=signal,
            confidence=confidence,
            probabilities={
                'DOWN': float(prob_down),
                'NEUTRAL': float(prob_neutral),
                'UP': float(prob_up),
            },
            raw_logits=logits.cpu().numpy()[0],
        )
    
    def predict_batch(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        x_strat: torch.Tensor,
    ) -> np.ndarray:
        """
        Run batch inference on pre-processed tensors.
        
        Args:
            x_fast: (B, 50, 15) tensor
            x_slow: (B, 20, 8) tensor
            x_strat: (B, 64) tensor
            
        Returns:
            (B, 3) array of probabilities
        """
        with torch.no_grad():
            x_fast = x_fast.to(self.device)
            x_slow = x_slow.to(self.device)
            x_strat = x_strat.to(self.device)
            
            logits = self.model(x_fast, x_slow, x_strat)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
        return probs


def create_adapter(
    model_path: str = "models/v5_ultimate/best_model.pt",
    scaler_path: str = "models/v5_ultimate/scaler_params.json",
) -> GoldenBreezeAdapter:
    """Factory function to create adapter with default paths."""
    return GoldenBreezeAdapter(model_path=model_path, scaler_path=scaler_path)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing GoldenBreezeAdapter")
    print("=" * 60)
    
    # Create adapter
    adapter = GoldenBreezeAdapter()
    
    # Create dummy data
    np.random.seed(42)
    n_m5 = 100
    n_h1 = 50
    
    # Simulate M5 data
    df_m5 = pd.DataFrame({
        'open': 1900 + np.random.randn(n_m5).cumsum() * 0.1,
        'high': 1900 + np.random.randn(n_m5).cumsum() * 0.1 + 0.5,
        'low': 1900 + np.random.randn(n_m5).cumsum() * 0.1 - 0.5,
        'close': 1900 + np.random.randn(n_m5).cumsum() * 0.1,
        'volume': np.random.randint(100, 1000, n_m5),
    })
    
    # Simulate H1 data
    df_h1 = pd.DataFrame({
        'open': 1900 + np.random.randn(n_h1).cumsum() * 0.5,
        'high': 1900 + np.random.randn(n_h1).cumsum() * 0.5 + 2,
        'low': 1900 + np.random.randn(n_h1).cumsum() * 0.5 - 2,
        'close': 1900 + np.random.randn(n_h1).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, n_h1),
    })
    
    # Run prediction
    print("\n🔮 Running prediction...")
    result = adapter.predict(df_m5, df_h1)
    
    print(f"\n📊 Result:")
    print(f"   Signal: {result.signal}")
    print(f"   Confidence: {result.confidence:.4f}")
    print(f"   Probabilities: DOWN={result.probabilities['DOWN']:.4f}, "
          f"NEUTRAL={result.probabilities['NEUTRAL']:.4f}, "
          f"UP={result.probabilities['UP']:.4f}")
    
    print("\n✅ Test passed!")

"""
Golden Breeze V4 - LSTM Inference Adapter

Provides a high-level interface for running predictions with the LSTM V4 model.
Handles feature generation, model loading, and prediction formatting.

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v4_lstm import LSTMModelV4, LSTMConfig
from aimodule.data_pipeline.strategy_signals import StrategySignalsGenerator


@dataclass
class PredictionResult:
    """Result of a prediction."""
    pred_class: int           # 0=DOWN, 1=NEUTRAL, 2=UP
    label: str                # 'DOWN', 'NEUTRAL', 'UP'
    probs: np.ndarray         # [p_down, p_neutral, p_up]
    confidence: float         # max probability
    timestamp: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        return {
            'class': self.pred_class,
            'label': self.label,
            'probs': self.probs.tolist(),
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class LSTMV4Adapter:
    """
    Inference adapter for LSTM V4 model.
    
    Handles:
    - Model loading from checkpoint
    - Feature generation (V3 features, SMC, Strategy signals)
    - Sequence windowing
    - Prediction with formatted output
    
    Example:
        >>> adapter = LSTMV4Adapter('models/v4_5class/lstm_3class_best.pt')
        >>> result = adapter.predict(df_m5, df_h1)
        >>> print(result.label)  # 'UP'
    """
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    # Default sequence lengths
    SEQ_LEN_FAST = 50   # M5 bars
    SEQ_LEN_SLOW = 20   # H1 bars
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        config: Optional[LSTMConfig] = None,
    ):
        """
        Initialize adapter.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            config: Model config (uses default if None)
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.config = config or LSTMConfig()
        self.model = self._load_model()
        
        # Initialize feature generators
        self.strategy_gen = StrategySignalsGenerator()
        
        print(f"✅ LSTMV4Adapter initialized on {self.device}")
    
    def _load_model(self) -> LSTMModelV4:
        """Load model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        model = LSTMModelV4(config=self.config)
        
        state_dict = torch.load(
            self.checkpoint_path, 
            map_location=self.device, 
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _generate_v3_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate V3 features from OHLCV data.
        
        Features (15 total):
        - OHLCV normalized (5)
        - Returns (1)
        - Volatility (1)
        - RSI (1)
        - MACD (3)
        - Bollinger (2)
        - ATR (1)
        - Volume MA ratio (1)
        """
        df = df.copy()
        
        # Normalize OHLCV by close
        close = df['close'].values
        df['open_norm'] = df['open'] / close
        df['high_norm'] = df['high'] / close
        df['low_norm'] = df['low'] / close
        df['close_norm'] = 1.0
        
        # Volume normalization
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        vol_ma = df[vol_col].rolling(20).mean()
        df['vol_norm'] = df[vol_col] / (vol_ma + 1e-8)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Volatility (20-period)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'] / 100  # Normalize to 0-1
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / close
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = (sma20 + 2 * std20) / close
        df['bb_lower'] = (sma20 - 2 * std20) / close
        
        # ATR
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean() / close
        
        # Volume MA ratio
        df['vol_ma_ratio'] = df[vol_col] / (df[vol_col].rolling(20).mean() + 1e-8)
        
        # Select features
        feature_cols = [
            'open_norm', 'high_norm', 'low_norm', 'close_norm', 'vol_norm',
            'returns', 'volatility', 'rsi',
            'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower',
            'atr', 'vol_ma_ratio',
        ]
        
        features = df[feature_cols].fillna(0).values.astype(np.float32)
        
        return features
    
    def _generate_smc_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate SMC features from H1 data.
        
        Features (8 total):
        - Trend direction (1)
        - Trend strength (1)
        - Support distance (1)
        - Resistance distance (1)
        - Volume profile (1)
        - Session (1)
        - Momentum (1)
        - Structure (1)
        """
        df = df.copy()
        close = df['close'].values
        
        # Trend direction (EMA20 vs EMA50)
        ema20 = df['close'].ewm(span=20).mean()
        ema50 = df['close'].ewm(span=50).mean()
        df['trend_dir'] = ((ema20 - ema50) / close).clip(-0.1, 0.1) * 10
        
        # Trend strength (ADX-like)
        high = df['high']
        low = df['low']
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = (high - low).rolling(14).mean()
        df['trend_strength'] = ((plus_dm - minus_dm).abs().rolling(14).mean() / (tr + 1e-8)).clip(0, 1)
        
        # Support distance (distance to 20-period low)
        low20 = df['low'].rolling(20).min()
        df['support_dist'] = ((close - low20) / close).clip(0, 0.1) * 10
        
        # Resistance distance (distance to 20-period high)
        high20 = df['high'].rolling(20).max()
        df['resist_dist'] = ((high20 - close) / close).clip(0, 0.1) * 10
        
        # Volume profile
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
        vol_ma = df[vol_col].rolling(20).mean()
        df['vol_profile'] = (df[vol_col] / (vol_ma + 1e-8)).clip(0, 3) / 3
        
        # Session encoding (simplified)
        if 'time' in df.columns:
            hours = pd.to_datetime(df['time']).dt.hour
            df['session'] = ((hours >= 8) & (hours < 16)).astype(float) * 0.5 + \
                           ((hours >= 13) & (hours < 22)).astype(float) * 0.5
        else:
            df['session'] = 0.5
        
        # Momentum (ROC)
        df['momentum'] = (df['close'].pct_change(5)).clip(-0.1, 0.1) * 10
        
        # Structure (higher highs / lower lows)
        hh = (df['high'] > df['high'].shift(1)).astype(float)
        ll = (df['low'] < df['low'].shift(1)).astype(float)
        df['structure'] = (hh.rolling(5).mean() - ll.rolling(5).mean()).clip(-1, 1)
        
        feature_cols = [
            'trend_dir', 'trend_strength', 'support_dist', 'resist_dist',
            'vol_profile', 'session', 'momentum', 'structure',
        ]
        
        features = df[feature_cols].fillna(0).values.astype(np.float32)
        
        return features
    
    def _create_windows(
        self,
        m5_features: np.ndarray,
        h1_features: np.ndarray,
        strategy_signals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows for prediction.
        
        Returns last valid window.
        """
        # Get last SEQ_LEN_FAST M5 bars
        if len(m5_features) < self.SEQ_LEN_FAST:
            # Pad with zeros
            pad = np.zeros((self.SEQ_LEN_FAST - len(m5_features), m5_features.shape[1]))
            x_fast = np.vstack([pad, m5_features])
        else:
            x_fast = m5_features[-self.SEQ_LEN_FAST:]
        
        # Get last SEQ_LEN_SLOW H1 bars
        if len(h1_features) < self.SEQ_LEN_SLOW:
            pad = np.zeros((self.SEQ_LEN_SLOW - len(h1_features), h1_features.shape[1]))
            x_slow = np.vstack([pad, h1_features])
        else:
            x_slow = h1_features[-self.SEQ_LEN_SLOW:]
        
        # Get last strategy signals
        x_strat = strategy_signals[-1] if len(strategy_signals) > 0 else np.zeros(64)
        
        return x_fast, x_slow, x_strat
    
    def predict(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> PredictionResult:
        """
        Run prediction on current market state.
        
        Args:
            df_m5: M5 OHLCV DataFrame (at least SEQ_LEN_FAST bars)
            df_h1: H1 OHLCV DataFrame (at least SEQ_LEN_SLOW bars)
            timestamp: Optional timestamp for result
            
        Returns:
            PredictionResult with class, label, probabilities
        """
        # Generate features
        m5_features = self._generate_v3_features(df_m5)
        h1_features = self._generate_smc_features(df_h1)
        
        # Generate strategy signals
        strategy_df = self.strategy_gen.generate_all_signals(df_m5)
        strategy_signals = strategy_df.values.astype(np.float32)
        
        # Create windows
        x_fast, x_slow, x_strat = self._create_windows(
            m5_features, h1_features, strategy_signals
        )
        
        # Convert to tensors
        x_fast_t = torch.tensor(x_fast, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_slow_t = torch.tensor(x_slow, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_strat_t = torch.tensor(x_strat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_fast_t, x_slow_t, x_strat_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(logits.argmax(dim=-1).cpu().item())
        
        return PredictionResult(
            pred_class=pred_class,
            label=self.CLASS_NAMES[pred_class],
            probs=probs,
            confidence=float(probs.max()),
            timestamp=timestamp,
        )
    
    def predict_batch(
        self,
        x_fast: np.ndarray,
        x_slow: np.ndarray,
        x_strat: np.ndarray,
    ) -> List[PredictionResult]:
        """
        Run batch prediction on pre-computed features.
        
        Args:
            x_fast: (batch, seq_fast, 15)
            x_slow: (batch, seq_slow, 8)
            x_strat: (batch, 64)
            
        Returns:
            List of PredictionResult
        """
        # Convert to tensors
        x_fast_t = torch.tensor(x_fast, dtype=torch.float32).to(self.device)
        x_slow_t = torch.tensor(x_slow, dtype=torch.float32).to(self.device)
        x_strat_t = torch.tensor(x_strat, dtype=torch.float32).to(self.device)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_fast_t, x_slow_t, x_strat_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_classes = logits.argmax(dim=-1).cpu().numpy()
        
        results = []
        for i in range(len(pred_classes)):
            results.append(PredictionResult(
                pred_class=int(pred_classes[i]),
                label=self.CLASS_NAMES[pred_classes[i]],
                probs=probs[i],
                confidence=float(probs[i].max()),
            ))
        
        return results
    
    def dummy_predict(self) -> Dict:
        """
        Run a dummy prediction to verify model loads correctly.
        
        Returns:
            dict with test results
        """
        # Create dummy inputs
        x_fast = torch.randn(1, self.SEQ_LEN_FAST, 15).to(self.device)
        x_slow = torch.randn(1, self.SEQ_LEN_SLOW, 8).to(self.device)
        x_strat = torch.randn(1, 64).to(self.device)
        
        # Run prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_fast, x_slow, x_strat)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(logits.argmax(dim=-1).cpu().item())
        
        return {
            'status': 'ok',
            'pred_class': pred_class,
            'label': self.CLASS_NAMES[pred_class],
            'probs': probs.tolist(),
            'device': str(self.device),
        }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("LSTMV4Adapter - Quick Test")
    print("=" * 60)
    
    checkpoint = "models/v4_5class/lstm_3class_best.pt"
    
    if os.path.exists(checkpoint):
        adapter = LSTMV4Adapter(checkpoint)
        result = adapter.dummy_predict()
        print(f"\nDummy prediction result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("\n✅ LSTMV4Adapter test passed!")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint}")
        print("Creating adapter without loading weights...")
        
        # Just test the structure
        from aimodule.models.v4_lstm import LSTMModelV4
        model = LSTMModelV4()
        print(f"Model params: {sum(p.numel() for p in model.parameters())}")

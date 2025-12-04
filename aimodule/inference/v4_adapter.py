"""
Golden Breeze v4 - Inference Adapter

Provides high-level inference interface for the Fusion Transformer v4 model.
Handles data preparation, SMC processing, time alignment, and prediction.

Usage:
    >>> adapter = V4InferenceAdapter("models/direction_transformer_v4_best_mcc.pt")
    >>> result = adapter.predict(df_m5, df_h1)
    >>> print(f"Score: {result['score']:.4f}, Class: {result['class']}, Confidence: {result['confidence']:.4f}")

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ..models.v4_transformer.model import GoldenBreezeFusionV4
from ..models.v4_transformer.config import V4Config
from ..data_pipeline.smc_processor import SMCProcessor
from ..data_pipeline.alignment import TimeAligner


class V4InferenceAdapter:
    """
    High-level inference adapter for GoldenBreezeFusionV4.
    
    This adapter handles:
    1. Model loading with config
    2. SMC feature processing from H1 data
    3. Time alignment between M5 and H1
    4. OHLCV normalization
    5. Batch-ready tensor preparation
    6. Inference and result formatting
    
    Key Design:
    - input_channels=5 (OHLCV only, no derived features)
    - Supports both single-sample and batch inference
    - Auto-detects volume column (tick_volume vs volume)
    
    Example:
        >>> adapter = V4InferenceAdapter("models/best_model.pt")
        >>> result = adapter.predict(df_m5, df_h1)
        >>> # result = {'score': 0.65, 'class': 1, 'confidence': 0.72, 'label': 'UP'}
    """
    
    # Class labels
    CLASS_LABELS = {0: 'DOWN', 1: 'HOLD', 2: 'UP'}
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        config: Optional[V4Config] = None,
    ):
        """
        Initialize the inference adapter.
        
        Args:
            model_path: Path to saved model checkpoint (.pt file)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            config: Optional V4Config override (loaded from checkpoint if None)
        """
        self.model_path = Path(model_path)
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.config = self._load_model(config)
        self.model.eval()
        
        # Initialize processors
        self.smc_processor = SMCProcessor(
            decay_lambda=self.config.smc_decay_lambda,
            max_ob_age=self.config.smc_max_ob_age,
            max_active_obs=self.config.max_dynamic_tokens,
        )
        self.time_aligner = TimeAligner()
        
        # Cache for normalization stats
        self._norm_stats = {}
        
        print(f"✅ V4InferenceAdapter initialized")
        print(f"   Model: {self.model_path.name}")
        print(f"   Device: {self.device}")
        print(f"   Seq lengths: M5={self.config.seq_len_fast}, H1={self.config.seq_len_slow}")
    
    def _load_model(
        self, 
        config_override: Optional[V4Config] = None
    ) -> Tuple[GoldenBreezeFusionV4, V4Config]:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get config
        if config_override:
            config = config_override
        elif 'config' in checkpoint:
            config = V4Config.from_dict(checkpoint['config'])
        else:
            print("⚠️  No config in checkpoint, using default")
            config = V4Config()
        
        # Create model
        model = GoldenBreezeFusionV4(config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume entire checkpoint is state_dict
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        return model, config
    
    @staticmethod
    def _get_volume_col(df: pd.DataFrame) -> str:
        """Detect volume column name."""
        if 'tick_volume' in df.columns:
            return 'tick_volume'
        elif 'volume' in df.columns:
            return 'volume'
        else:
            raise ValueError(f"No volume column found. Columns: {list(df.columns)}")
    
    def _prepare_ohlcv(
        self,
        df: pd.DataFrame,
        seq_len: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Prepare OHLCV data as numpy array.
        
        Args:
            df: DataFrame with OHLCV data
            seq_len: Number of bars to extract
            normalize: Whether to normalize values
            
        Returns:
            Array of shape (seq_len, 5) with [O, H, L, C, V]
        """
        # Get last seq_len bars
        if len(df) < seq_len:
            raise ValueError(f"Not enough data: {len(df)} < {seq_len}")
        
        df_slice = df.iloc[-seq_len:].copy()
        
        # Detect volume column
        vol_col = self._get_volume_col(df_slice)
        
        # Extract OHLCV
        ohlcv = df_slice[['open', 'high', 'low', 'close', vol_col]].values.astype(np.float32)
        
        if normalize:
            # Normalize OHLC by close price (relative changes)
            close_mean = ohlcv[:, 3].mean()
            if close_mean > 0:
                ohlcv[:, :4] = (ohlcv[:, :4] - close_mean) / close_mean * 100
            
            # Normalize volume by rolling mean
            vol_mean = ohlcv[:, 4].mean()
            if vol_mean > 0:
                ohlcv[:, 4] = ohlcv[:, 4] / vol_mean - 1
        
        return ohlcv
    
    def _prepare_smc_features(
        self,
        df_h1: pd.DataFrame,
        aligned_h1_time: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare SMC features for the aligned H1 time.
        
        Returns:
            (static_smc, dynamic_smc) arrays
        """
        # Process H1 data
        smc_features = self.smc_processor.process_h1_data(df_h1)
        
        # Find the row for aligned_h1_time
        if aligned_h1_time in smc_features.index:
            row = smc_features.loc[aligned_h1_time]
        else:
            # Find nearest
            valid_times = smc_features.index[smc_features.index <= aligned_h1_time]
            if len(valid_times) > 0:
                row = smc_features.loc[valid_times[-1]]
            else:
                # Return zeros if no valid H1
                static = np.zeros(self.config.static_smc_dim, dtype=np.float32)
                dynamic = np.zeros((self.config.max_dynamic_tokens, self.config.dynamic_smc_dim), dtype=np.float32)
                return static, dynamic
        
        # Extract vectors
        static = self.smc_processor.get_static_vector(row, self.config.static_smc_dim)
        
        dynamic_obs = row.get('dynamic_obs', []) if hasattr(row, 'get') else row['dynamic_obs'] if 'dynamic_obs' in row.index else []
        dynamic = self.smc_processor.get_dynamic_matrix(
            dynamic_obs,
            max_tokens=self.config.max_dynamic_tokens,
            dim_per_token=self.config.dynamic_smc_dim,
        )
        
        return static, dynamic
    
    def predict(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        return_raw: bool = False,
    ) -> Dict[str, Union[float, int, str, np.ndarray]]:
        """
        Run inference on M5 + H1 data.
        
        Args:
            df_m5: M5 OHLCV DataFrame (needs at least seq_len_fast bars)
            df_h1: H1 OHLCV DataFrame (needs at least seq_len_slow bars)
            return_raw: If True, include raw model outputs
            
        Returns:
            Dictionary with:
            - score: float in [-1, 1] (continuous prediction)
            - class: int (0=DOWN, 1=HOLD, 2=UP)
            - confidence: float in [0, 1]
            - label: str ('DOWN', 'HOLD', 'UP')
            - probs: Optional[ndarray] if return_raw
        """
        # Validate input
        if len(df_m5) < self.config.seq_len_fast:
            raise ValueError(f"M5 data too short: {len(df_m5)} < {self.config.seq_len_fast}")
        if len(df_h1) < self.config.seq_len_slow:
            raise ValueError(f"H1 data too short: {len(df_h1)} < {self.config.seq_len_slow}")
        
        # Get current M5 time
        if 'time' in df_m5.columns:
            m5_time = pd.to_datetime(df_m5['time'].iloc[-1])
        else:
            m5_time = pd.to_datetime(df_m5.index[-1])
        
        # Align to get H1 time
        aligned_h1_time = self.time_aligner.get_last_closed_h1(m5_time)
        
        # Prepare M5 window
        m5_window = self._prepare_ohlcv(df_m5, self.config.seq_len_fast)
        
        # Prepare H1 window (filter to aligned time)
        if 'time' in df_h1.columns:
            h1_mask = pd.to_datetime(df_h1['time']) <= aligned_h1_time
        else:
            h1_mask = pd.to_datetime(df_h1.index) <= aligned_h1_time
        
        df_h1_filtered = df_h1[h1_mask]
        
        if len(df_h1_filtered) < self.config.seq_len_slow:
            # Not enough H1 data, use what we have
            print(f"⚠️  Limited H1 data: {len(df_h1_filtered)} bars")
            df_h1_filtered = df_h1.iloc[-self.config.seq_len_slow:]
        
        h1_window = self._prepare_ohlcv(df_h1_filtered, self.config.seq_len_slow)
        
        # Prepare SMC features
        static_smc, dynamic_smc = self._prepare_smc_features(df_h1_filtered, aligned_h1_time)
        
        # Convert to tensors (batch size = 1)
        x_fast = torch.from_numpy(m5_window).unsqueeze(0).to(self.device)  # (1, 200, 5)
        x_slow = torch.from_numpy(h1_window).unsqueeze(0).to(self.device)  # (1, 50, 5)
        smc_static = torch.from_numpy(static_smc).unsqueeze(0).to(self.device)  # (1, 16)
        smc_dynamic = torch.from_numpy(dynamic_smc).unsqueeze(0).to(self.device)  # (1, 10, 12)
        
        # Run inference
        with torch.no_grad():
            logits, score = self.model(x_fast, x_slow, smc_static, smc_dynamic)
            
            # Get class prediction
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
            
            # Get score
            score_val = score.item()
        
        # Format result
        result = {
            'score': float(np.clip(score_val, -1, 1)),
            'class': int(pred_class),
            'confidence': float(confidence),
            'label': self.CLASS_LABELS[pred_class],
            'aligned_h1_time': aligned_h1_time,
        }
        
        if return_raw:
            result['probs'] = probs.cpu().numpy()[0]
            result['logits'] = logits.cpu().numpy()[0]
        
        return result
    
    def predict_batch(
        self,
        m5_windows: np.ndarray,
        h1_windows: np.ndarray,
        smc_static_batch: np.ndarray,
        smc_dynamic_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Batch inference for multiple samples.
        
        Args:
            m5_windows: (batch, seq_len_fast, 5)
            h1_windows: (batch, seq_len_slow, 5)
            smc_static_batch: (batch, static_smc_dim)
            smc_dynamic_batch: (batch, max_tokens, dynamic_smc_dim)
            
        Returns:
            Dictionary with batch results:
            - scores: (batch,) array
            - classes: (batch,) array
            - confidences: (batch,) array
            - probs: (batch, num_classes) array
        """
        # Convert to tensors
        x_fast = torch.from_numpy(m5_windows).float().to(self.device)
        x_slow = torch.from_numpy(h1_windows).float().to(self.device)
        smc_static = torch.from_numpy(smc_static_batch).float().to(self.device)
        smc_dynamic = torch.from_numpy(smc_dynamic_batch).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits, scores = self.model(x_fast, x_slow, smc_static, smc_dynamic)
            probs = torch.softmax(logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)
            confidences = probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        return {
            'scores': scores.cpu().numpy().flatten(),
            'classes': pred_classes.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'probs': probs.cpu().numpy(),
        }
    
    def get_signal(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        score_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Get trading signal with thresholds.
        
        Args:
            df_m5: M5 OHLCV data
            df_h1: H1 OHLCV data
            score_threshold: Minimum absolute score for signal
            confidence_threshold: Minimum confidence for signal
            
        Returns:
            Dictionary with:
            - signal: 'BUY', 'SELL', or 'HOLD'
            - strength: float (0-1)
            - is_valid: bool (whether thresholds are met)
            - details: prediction details
        """
        pred = self.predict(df_m5, df_h1)
        
        score = pred['score']
        confidence = pred['confidence']
        
        # Determine signal
        if abs(score) < score_threshold or confidence < confidence_threshold:
            signal = 'HOLD'
            is_valid = False
        elif score > score_threshold:
            signal = 'BUY'
            is_valid = True
        else:
            signal = 'SELL'
            is_valid = True
        
        return {
            'signal': signal,
            'strength': abs(score),
            'is_valid': is_valid,
            'confidence': confidence,
            'details': pred,
        }


# === Convenience function ===

def load_v4_adapter(model_path: str = None) -> V4InferenceAdapter:
    """
    Load V4InferenceAdapter with default path discovery.
    
    Searches for model in:
    1. Provided path
    2. models/direction_transformer_v4_best_mcc_*.pt
    3. models/direction_transformer_v4_best_loss_*.pt
    """
    from pathlib import Path
    
    if model_path and Path(model_path).exists():
        return V4InferenceAdapter(model_path)
    
    # Search for models
    model_dir = Path("models")
    patterns = [
        "direction_transformer_v4_best_mcc_*.pt",
        "direction_transformer_v4_best_loss_*.pt",
        "direction_transformer_v4_*.pt",
    ]
    
    for pattern in patterns:
        matches = list(model_dir.glob(pattern))
        if matches:
            # Use most recent
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return V4InferenceAdapter(str(matches[0]))
    
    raise FileNotFoundError(f"No v4 model found in {model_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("V4InferenceAdapter - Quick Test")
    print("=" * 60)
    
    # Try to load adapter
    try:
        adapter = load_v4_adapter()
        print(f"\n✅ Adapter loaded successfully!")
        print(f"   Model parameters: {sum(p.numel() for p in adapter.model.parameters()):,}")
        
        # Create dummy data for testing
        import numpy as np
        
        n_m5 = 250
        n_h1 = 60
        
        df_m5 = pd.DataFrame({
            'time': pd.date_range(end=pd.Timestamp.now(), periods=n_m5, freq='5min'),
            'open': 2650 + np.random.randn(n_m5).cumsum(),
            'high': 2655 + np.random.randn(n_m5).cumsum(),
            'low': 2645 + np.random.randn(n_m5).cumsum(),
            'close': 2650 + np.random.randn(n_m5).cumsum(),
            'tick_volume': np.random.randint(100, 1000, n_m5),
        })
        
        df_h1 = pd.DataFrame({
            'time': pd.date_range(end=pd.Timestamp.now(), periods=n_h1, freq='h'),
            'open': 2650 + np.random.randn(n_h1).cumsum() * 5,
            'high': 2655 + np.random.randn(n_h1).cumsum() * 5,
            'low': 2645 + np.random.randn(n_h1).cumsum() * 5,
            'close': 2650 + np.random.randn(n_h1).cumsum() * 5,
            'volume': np.random.randint(1000, 10000, n_h1),
        })
        
        print(f"\nTest data: M5={len(df_m5)} bars, H1={len(df_h1)} bars")
        
        # Run prediction
        result = adapter.predict(df_m5, df_h1, return_raw=True)
        
        print(f"\nPrediction:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Class: {result['class']} ({result['label']})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probs: {result['probs']}")
        
        # Test signal
        signal = adapter.get_signal(df_m5, df_h1)
        print(f"\nSignal: {signal['signal']} (strength={signal['strength']:.4f}, valid={signal['is_valid']})")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Run training first to create a model checkpoint.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

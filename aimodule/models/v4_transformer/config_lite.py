"""
Golden Breeze v4 Lite - Simplified Config

Упрощённая конфигурация, более похожая на v3:
- Меньше параметров модели
- Готовые признаки вместо сырых OHLCV
- Простой CrossEntropyLoss вместо FocalLoss

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class V4LiteConfig:
    """
    v4 Lite Configuration - Simplified for Better Training
    
    Changes from v4:
    - Uses engineered features (like v3) instead of raw OHLCV
    - Smaller model (fewer parameters)
    - CrossEntropyLoss instead of FocalLoss
    - Adam instead of AdamW
    - Constant LR instead of OneCycleLR
    """
    
    # === Model Architecture ===
    # Reduced dimensions for faster training
    d_model: int = 64  # v4: 128 → v4_lite: 64 (like v3's hidden_size)
    nhead: int = 4  # v4: 4 → v4_lite: 4 (same)
    num_layers: int = 2  # v4: 2 → v4_lite: 2 (same)
    dim_feedforward: int = 128  # v4: 256 → v4_lite: 128
    dropout: float = 0.2  # v4: 0.1 → v4_lite: 0.2 (like v3)
    
    # === Sequence Lengths ===
    seq_len_fast: int = 50  # v4: 200 → v4_lite: 50 (like v3)
    seq_len_slow: int = 50  # Same as v3
    
    # === Feature Configuration ===
    # Use engineered features (like v3) instead of raw OHLCV
    # v3 uses 15 features: close, returns, log_returns, sma_fast, sma_slow, sma_ratio,
    #                      atr, atr_norm, rsi, bb_position, volume_ratio,
    #                      SMC_FVG_Bullish, SMC_FVG_Bearish, SMC_Swing_High, SMC_Swing_Low
    input_channels: int = 15  # v4: 5 → v4_lite: 15 (engineered features)
    
    # === SMC Features ===
    # Simplified - use only static SMC, no dynamic OB tokens
    static_smc_dim: int = 8  # v4: 16 → v4_lite: 8 (simpler)
    use_dynamic_smc: bool = False  # v4: True → v4_lite: False (simpler)
    dynamic_smc_dim: int = 0  # Not used
    max_dynamic_tokens: int = 0  # Not used
    
    # === Strategy Signals (from Phase 5) ===
    use_strategy_signals: bool = True
    strategy_signal_dim: int = 33  # 9 strategies × various signals
    
    # === Output ===
    num_classes: int = 2  # 0=DOWN, 1=UP (like v3 - no HOLD for simplicity)
    
    # === Training (v3-like parameters) ===
    batch_size: int = 64  # v4: 32 → v4_lite: 64 (like v3)
    learning_rate: float = 1e-3  # v4: 1e-4 → v4_lite: 1e-3 (like v3)
    weight_decay: float = 0.0  # v4: 0.01 → v4_lite: 0 (like v3's Adam)
    
    # Simple training (no warmup/cyclic)
    use_warmup: bool = False  # v4: True → v4_lite: False
    
    # Early stopping (like v3)
    patience: int = 10  # Increased from 5
    min_delta: float = 0.001
    
    # Loss function
    use_focal_loss: bool = False  # Use simple CrossEntropyLoss like v3
    label_smoothing: float = 0.0  # No smoothing
    
    # Epochs
    epochs: int = 50  # Same as v3
    
    # === Device ===
    device: str = "cuda"
    
    # === Feature Engineering ===
    # List of engineered features (like v3)
    m5_feature_list: List[str] = field(default_factory=lambda: [
        'close', 'returns', 'log_returns',
        'sma_fast', 'sma_slow', 'sma_ratio',
        'atr', 'atr_norm',
        'rsi', 'bb_position',
        'volume_ratio',
        'SMC_FVG_Bullish', 'SMC_FVG_Bearish',
        'SMC_Swing_High', 'SMC_Swing_Low'
    ])
    
    def __post_init__(self):
        """Validate and compute derived values."""
        # Ensure dimensions are compatible
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        
        # Set dynamic SMC to 0 if not used
        if not self.use_dynamic_smc:
            self.dynamic_smc_dim = 0
            self.max_dynamic_tokens = 0
    
    def get_model_dim(self) -> int:
        """Total model dimension for pooling."""
        return self.d_model * 2  # fast + slow streams
    
    def get_total_params_estimate(self) -> int:
        """Rough estimate of total model parameters."""
        # Embedding layers
        embed = self.input_channels * self.d_model * 2  # fast + slow
        smc_embed = self.static_smc_dim * self.d_model
        
        # Transformer layers (per layer)
        attn = self.d_model * self.d_model * 4 * self.num_layers  # Q, K, V, O
        ff = self.d_model * self.dim_feedforward * 2 * self.num_layers
        
        # Classification head
        head = (self.d_model * 2) * self.d_model + self.d_model * self.num_classes
        
        # Strategy signals
        strat = self.strategy_signal_dim * self.d_model if self.use_strategy_signals else 0
        
        return embed + smc_embed + attn + ff + head + strat
    
    def summary(self) -> str:
        """Print config summary."""
        lines = [
            "=" * 60,
            "Golden Breeze v4 Lite Configuration",
            "=" * 60,
            "",
            "🏗️ Architecture:",
            f"   d_model: {self.d_model}",
            f"   nhead: {self.nhead}",
            f"   num_layers: {self.num_layers}",
            f"   dim_feedforward: {self.dim_feedforward}",
            f"   dropout: {self.dropout}",
            "",
            "📊 Features:",
            f"   input_channels: {self.input_channels} (engineered features like v3)",
            f"   seq_len_fast (M5): {self.seq_len_fast}",
            f"   seq_len_slow (H1): {self.seq_len_slow}",
            f"   static_smc_dim: {self.static_smc_dim}",
            f"   use_dynamic_smc: {self.use_dynamic_smc}",
            f"   strategy_signals: {self.strategy_signal_dim if self.use_strategy_signals else 'disabled'}",
            "",
            "🎯 Training (v3-like):",
            f"   batch_size: {self.batch_size}",
            f"   learning_rate: {self.learning_rate}",
            f"   weight_decay: {self.weight_decay}",
            f"   use_warmup: {self.use_warmup}",
            f"   use_focal_loss: {self.use_focal_loss}",
            f"   patience: {self.patience}",
            f"   epochs: {self.epochs}",
            "",
            f"📐 Estimated Parameters: ~{self.get_total_params_estimate():,}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    config = V4LiteConfig()
    print(config.summary())

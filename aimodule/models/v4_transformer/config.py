"""
Golden Breeze Fusion Transformer v4.0 - Configuration

Contains all hyperparameters and architecture settings for the v4 model.

Author: Golden Breeze Team
Version: 4.0.0
"""

from dataclasses import dataclass, field
from typing import List, Optional


# === SMC Constants ===
SMC_DECAY_LAMBDA: float = 0.05  # Exponential decay rate for OB age
SMC_MAX_OB_AGE_BARS: int = 200  # Maximum age in bars before OB is considered expired
SMC_STATE_FRESH: int = 0
SMC_STATE_MITIGATED: int = 1
SMC_STATE_BROKEN: int = 2
SMC_NUM_STATES: int = 3


@dataclass
class V4Config:
    """
    Configuration for GoldenBreezeFusionV4 Transformer.
    
    Architecture Parameters:
    - d_model: Transformer hidden dimension
    - nhead: Number of attention heads
    - num_layers_fast: Encoder layers for fast stream (M5)
    - num_layers_slow: Encoder layers for slow stream (H1)
    
    Patch Encoding:
    - patch_size: Size of sliding window patch
    - patch_stride: Stride between patches (overlap = patch_size - stride)
    - input_channels: Number of OHLCV features (5 for OHLCV, 6 with volume normalized)
    
    SMC Features:
    - static_smc_dim: Dimension of static SMC features
      [rel_high, rel_low, is_bullish, time_decay, state_fresh, state_mitigated, state_broken,
       market_structure_score, liquidity_zone_proximity, session_bias, trend_bias, ...]
    - dynamic_smc_dim: Dimension of dynamic SMC events (per OB/FVG token)
    - max_dynamic_tokens: Maximum number of active OB/FVG tokens
    
    Output Heads:
    - num_classes: Number of output classes (UP, DOWN, HOLD)
    - score_hidden_dim: Hidden dimension for continuous score head
    """
    
    # === Transformer Architecture ===
    d_model: int = 128
    nhead: int = 4
    num_layers_fast: int = 2
    num_layers_slow: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    
    # === Sliding Patch Encoding ===
    patch_size: int = 16
    patch_stride: int = 8
    input_channels: int = 5  # OHLCV (Open, High, Low, Close, Volume)
    
    # === SMC Features (Updated for Phase 2) ===
    # Static SMC Vector: [rel_high, rel_low, is_bullish, time_decay, 
    #                     state_one_hot(3), market_structure(4), session(3), trend_bias(1)]
    static_smc_dim: int = 16  # Total static features per OB
    
    # Dynamic SMC: per-OB token features
    # [rel_high, rel_low, is_bullish, time_decay, state_embed(3), strength, type_embed(2)]
    dynamic_smc_dim: int = 12  # Features per dynamic OB token
    
    max_dynamic_tokens: int = 10  # Max active OBs to track
    
    # === SMC Decay Parameters ===
    smc_decay_lambda: float = SMC_DECAY_LAMBDA
    smc_max_ob_age: int = SMC_MAX_OB_AGE_BARS
    
    # === Sequence Lengths ===
    seq_len_fast: int = 200   # M5 bars (200 * 5min = 16.6 hours)
    seq_len_slow: int = 50    # H1 bars (50 * 1hour = 50 hours)
    
    # === Output Heads ===
    num_classes: int = 3      # UP, DOWN, HOLD
    score_hidden_dim: int = 64
    
    # === Training ===
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 10
    
    # === Regularization ===
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # === Device ===
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert self.patch_stride <= self.patch_size, \
            f"patch_stride ({self.patch_stride}) must be <= patch_size ({self.patch_size})"
        assert self.num_classes >= 2, \
            f"num_classes must be >= 2, got {self.num_classes}"
    
    @property
    def num_patches_fast(self) -> int:
        """Calculate number of patches for fast stream."""
        return (self.seq_len_fast - self.patch_size) // self.patch_stride + 1
    
    @property
    def num_patches_slow(self) -> int:
        """Calculate number of patches for slow stream."""
        return (self.seq_len_slow - self.patch_size) // self.patch_stride + 1
    
    @property
    def patch_dim(self) -> int:
        """Dimension of flattened patch (patch_size * input_channels)."""
        return self.patch_size * self.input_channels
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers_fast": self.num_layers_fast,
            "num_layers_slow": self.num_layers_slow,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "input_channels": self.input_channels,
            "static_smc_dim": self.static_smc_dim,
            "dynamic_smc_dim": self.dynamic_smc_dim,
            "max_dynamic_tokens": self.max_dynamic_tokens,
            "smc_decay_lambda": self.smc_decay_lambda,
            "smc_max_ob_age": self.smc_max_ob_age,
            "seq_len_fast": self.seq_len_fast,
            "seq_len_slow": self.seq_len_slow,
            "num_classes": self.num_classes,
            "score_hidden_dim": self.score_hidden_dim,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "V4Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# === Preset Configurations ===

def get_config_small() -> V4Config:
    """Small configuration for quick experiments."""
    return V4Config(
        d_model=64,
        nhead=2,
        num_layers_fast=1,
        num_layers_slow=1,
        dim_feedforward=256,
        seq_len_fast=100,
        seq_len_slow=25,
    )


def get_config_base() -> V4Config:
    """Base configuration (default)."""
    return V4Config()


def get_config_large() -> V4Config:
    """Large configuration for maximum accuracy."""
    return V4Config(
        d_model=256,
        nhead=8,
        num_layers_fast=4,
        num_layers_slow=4,
        dim_feedforward=1024,
        seq_len_fast=300,
        seq_len_slow=100,
        dropout=0.15,
    )

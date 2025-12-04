"""
Golden Breeze V4 - LSTM Model for 3-Class Direction Prediction

This is the winning architecture from Phase 3 experiments.
MCC: +0.12 (val) / +0.10 (test)

Classes:
    0: DOWN (price decrease)
    1: NEUTRAL (sideways)
    2: UP (price increase)

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """Configuration for LSTM V4 model."""
    
    # Input dimensions
    fast_features: int = 15      # M5 input features (V3 features)
    slow_features: int = 8       # H1 input features (SMC features)
    strategy_dim: int = 64       # Strategy signals dimension
    
    # LSTM dimensions
    fast_hidden: int = 32        # Fast LSTM hidden size
    slow_hidden: int = 16        # Slow LSTM hidden size
    
    # MLP dimensions
    strategy_hidden: int = 32    # Strategy projection output
    head_hidden: int = 64        # Classification head hidden
    
    # Regularization
    dropout: float = 0.3
    
    # Output
    num_classes: int = 3         # DOWN, NEUTRAL, UP
    
    @property
    def fusion_dim(self) -> int:
        """Total dimension after fusion."""
        # BiLSTM doubles hidden size
        return (self.fast_hidden * 2) + (self.slow_hidden * 2) + self.strategy_hidden


class LSTMModelV4(nn.Module):
    """
    Bidirectional LSTM model for 3-class direction prediction.
    
    Architecture:
        Fast Stream (M5):
            - BiLSTM(15 -> 32) -> concat(h_fwd, h_bwd) -> 64
        
        Slow Stream (H1):
            - BiLSTM(8 -> 16) -> concat(h_fwd, h_bwd) -> 32
        
        Strategy:
            - Linear(64 -> 32) -> ReLU -> Dropout -> 32
        
        Fusion:
            - Concat(64 + 32 + 32) = 128
            - Linear(128 -> 64) -> ReLU -> Dropout
            - Linear(64 -> 3)
    
    Example:
        >>> model = LSTMModelV4()
        >>> x_fast = torch.randn(32, 50, 15)  # (batch, seq, features)
        >>> x_slow = torch.randn(32, 20, 8)
        >>> x_strat = torch.randn(32, 64)
        >>> logits = model(x_fast, x_slow, x_strat)
        >>> print(logits.shape)  # (32, 3)
    """
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        
        self.config = config or LSTMConfig()
        
        # Fast LSTM: (B, seq, 15) -> (B, 64)
        self.fast_lstm = nn.LSTM(
            input_size=self.config.fast_features,
            hidden_size=self.config.fast_hidden,
            batch_first=True,
            bidirectional=True,
        )
        
        # Slow LSTM: (B, seq, 8) -> (B, 32)
        self.slow_lstm = nn.LSTM(
            input_size=self.config.slow_features,
            hidden_size=self.config.slow_hidden,
            batch_first=True,
            bidirectional=True,
        )
        
        # Strategy projection: 64 -> 32
        self.strat_proj = nn.Sequential(
            nn.Linear(self.config.strategy_dim, self.config.strategy_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )
        
        # Classification head: 128 -> 3
        self.head = nn.Sequential(
            nn.Linear(self.config.fusion_dim, self.config.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden, self.config.num_classes),
        )
    
    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        x_strat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_fast: M5 sequence (B, seq_fast, 15)
            x_slow: H1 sequence (B, seq_slow, 8)
            x_strat: Strategy signals (B, 64)
            
        Returns:
            logits: (B, 3) class logits
        """
        # Fast stream: BiLSTM
        _, (h_fast, _) = self.fast_lstm(x_fast)
        # h_fast: (2, B, hidden) -> concat -> (B, hidden*2)
        h_fast = torch.cat([h_fast[0], h_fast[1]], dim=1)
        
        # Slow stream: BiLSTM
        _, (h_slow, _) = self.slow_lstm(x_slow)
        h_slow = torch.cat([h_slow[0], h_slow[1]], dim=1)
        
        # Strategy projection
        st = self.strat_proj(x_strat)
        
        # Fusion
        fused = torch.cat([h_fast, h_slow, st], dim=1)
        
        # Classification
        logits = self.head(fused)
        
        return logits
    
    def predict(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        x_strat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with probabilities.
        
        Returns:
            dict with 'logits', 'probs', 'pred_class'
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_fast, x_slow, x_strat)
            probs = F.softmax(logits, dim=-1)
            pred_class = logits.argmax(dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'pred_class': pred_class,
        }
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        return self.CLASS_NAMES[class_idx]
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[LSTMConfig] = None,
        device: str = 'cuda',
    ) -> 'LSTMModelV4':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt file
            config: Model config (uses default if None)
            device: Device to load to
            
        Returns:
            Loaded model
        """
        model = cls(config=config)
        
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model


# Backward compatibility alias
LSTMModel = LSTMModelV4


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("LSTMModelV4 - Quick Test")
    print("=" * 60)
    
    config = LSTMConfig()
    model = LSTMModelV4(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    x_fast = torch.randn(batch_size, 50, 15)
    x_slow = torch.randn(batch_size, 20, 8)
    x_strat = torch.randn(batch_size, 64)
    
    logits = model(x_fast, x_slow, x_strat)
    print(f"\nInput shapes:")
    print(f"  x_fast: {x_fast.shape}")
    print(f"  x_slow: {x_slow.shape}")
    print(f"  x_strat: {x_strat.shape}")
    print(f"\nOutput shape: {logits.shape}")
    
    # Test predict
    result = model.predict(x_fast, x_slow, x_strat)
    print(f"\nPrediction:")
    print(f"  Probs: {result['probs'][0].numpy()}")
    print(f"  Class: {result['pred_class'][0].item()} ({model.get_class_name(result['pred_class'][0].item())})")
    
    print("\n✅ LSTMModelV4 test passed!")

"""
Golden Breeze v4 Lite Model

Simplified Transformer model that uses engineered features like v3:
- Instead of learning from raw OHLCV, uses precomputed indicators
- Smaller model with fewer parameters
- More similar to v3's successful approach

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

from .config_lite import V4LiteConfig


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SimpleTransformerEncoder(nn.Module):
    """
    Simplified single-stream Transformer encoder.
    
    Unlike v4's dual-stream, this uses a single stream with
    engineered features as input.
    """
    
    def __init__(self, config: V4LiteConfig):
        super().__init__()
        self.config = config
        
        # Input projection from features to d_model
        self.input_proj = nn.Linear(config.input_channels, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.d_model,
            max_len=max(config.seq_len_fast, config.seq_len_slow) + 10,
            dropout=config.dropout,
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_channels)
        Returns:
            encoded: (batch, seq_len, d_model)
        """
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        return x


class GoldenBreezeLite(nn.Module):
    """
    Golden Breeze v4 Lite Model
    
    A simplified version of GoldenBreezeFusionV4 that:
    1. Uses engineered features (like v3) instead of raw OHLCV
    2. Has a single Transformer stream instead of dual-stream
    3. Optionally includes SMC static features
    4. Optionally includes strategy signals
    
    This is closer to v3's successful approach while using Transformer architecture.
    """
    
    def __init__(self, config: V4LiteConfig):
        super().__init__()
        self.config = config
        
        # Main Transformer encoder for M5 features
        self.encoder = SimpleTransformerEncoder(config)
        
        # Optional SMC static embedding
        if config.static_smc_dim > 0:
            self.smc_embed = nn.Sequential(
                nn.Linear(config.static_smc_dim, config.d_model),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.smc_embed = None
        
        # Optional Strategy signals embedding
        if config.use_strategy_signals and config.strategy_signal_dim > 0:
            self.strategy_embed = nn.Sequential(
                nn.Linear(config.strategy_signal_dim, config.d_model),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.strategy_embed = None
        
        # Calculate pooled dimension
        # Base: d_model from last timestep
        # + d_model if SMC static
        # + d_model if strategy signals
        pooled_dim = config.d_model
        if config.static_smc_dim > 0:
            pooled_dim += config.d_model
        if config.use_strategy_signals and config.strategy_signal_dim > 0:
            pooled_dim += config.d_model
        
        # Classification head (two-layer like v3)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        smc_static: Optional[torch.Tensor] = None,
        strategy_signals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_channels) - engineered features
            smc_static: (batch, static_smc_dim) - optional SMC features
            strategy_signals: (batch, strategy_signal_dim) - optional strategy signals
            
        Returns:
            logits: (batch, num_classes)
        """
        # Encode sequence
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        # Take last timestep (like v3's LSTM output)
        pooled = encoded[:, -1, :]  # (batch, d_model)
        
        # Add SMC static features if available
        if self.smc_embed is not None and smc_static is not None:
            smc_encoded = self.smc_embed(smc_static)  # (batch, d_model)
            pooled = torch.cat([pooled, smc_encoded], dim=-1)
        
        # Add strategy signals if available
        if self.strategy_embed is not None and strategy_signals is not None:
            strat_encoded = self.strategy_embed(strategy_signals)  # (batch, d_model)
            pooled = torch.cat([pooled, strat_encoded], dim=-1)
        
        # Classify
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits
    
    def predict(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and probabilities.
        
        Returns:
            predictions: (batch,) class predictions
            probabilities: (batch, num_classes) softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, **kwargs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs


if __name__ == "__main__":
    print("=" * 60)
    print("GoldenBreezeLite - Model Test")
    print("=" * 60)
    
    config = V4LiteConfig()
    print(config.summary())
    
    model = GoldenBreezeLite(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, config.seq_len_fast, config.input_channels)
    smc_static = torch.randn(batch_size, config.static_smc_dim)
    strategy_signals = torch.randn(batch_size, config.strategy_signal_dim)
    
    print(f"\n🧪 Forward pass test:")
    print(f"   x: {x.shape}")
    print(f"   smc_static: {smc_static.shape}")
    print(f"   strategy_signals: {strategy_signals.shape}")
    
    logits = model(x, smc_static, strategy_signals)
    print(f"   output: {logits.shape}")
    
    # Test predict
    preds, probs = model.predict(x, smc_static=smc_static, strategy_signals=strategy_signals)
    print(f"   predictions: {preds.shape}")
    print(f"   probabilities: {probs.shape}")
    
    print("\n✅ GoldenBreezeLite test passed!")

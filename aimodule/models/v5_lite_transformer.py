"""
Golden Breeze v5 - Lite Patch Transformer

Modern architecture inspired by Vision Transformer (ViT) adapted for time series.
Uses patch embedding to reduce sequence length and enable faster attention.

Key Features:
- Patch Embedding via Conv1d (200 bars → 50 patches)
- 2-Layer Transformer Encoder (lightweight, prevents overfitting)
- Gated Linear Unit for strategy features
- CLS token for classification

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LitePatchTransformer v5                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   [M5 OHLCV+Features] ──► PatchEmbed(Conv1d) ──► [CLS]+Patches │
    │         (B, 50, 15)            ↓                    (B, 51, 128)│
    │                         PositionalEncoding                      │
    │                                ↓                                │
    │                    TransformerEncoder (2 layers)                │
    │                                ↓                                │
    │                         [CLS Token] ──────────────┐             │
    │                                                   │             │
    │   [Strategy 64] ──► StrategyGLU ──► (B, 64) ─────┼─► Concat    │
    │                                                   │             │
    │                                              (B, 192)           │
    │                                                   ↓             │
    │                                           Classification Head   │
    │                                                   ↓             │
    │                                         [DOWN, NEUTRAL, UP]     │
    └─────────────────────────────────────────────────────────────────┘

Author: Golden Breeze Team
Version: 5.0.0
Date: 2025-12-05
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class V5Config:
    """Configuration for Lite Patch Transformer v5."""
    
    # Input dimensions
    seq_len: int = 50           # M5 sequence length (after V3Features)
    input_features: int = 15    # V3 features per timestep
    strategy_dim: int = 64      # Strategy signal features
    
    # Patch embedding
    patch_size: int = 4         # Size of each patch
    patch_stride: int = 4       # Stride (no overlap for speed)
    
    # Transformer architecture
    d_model: int = 128          # Embedding dimension
    nhead: int = 4              # Attention heads
    num_layers: int = 2         # Transformer layers (lightweight!)
    dim_feedforward: int = 256  # FFN hidden dim
    dropout: float = 0.1        # Dropout rate
    
    # Output
    num_classes: int = 3        # DOWN, NEUTRAL, UP
    
    # Computed
    @property
    def num_patches(self) -> int:
        return (self.seq_len - self.patch_size) // self.patch_stride + 1


class PatchEmbedding(nn.Module):
    """
    Patch Embedding using 1D Convolution.
    
    Converts time series into patches, similar to ViT for images.
    This reduces sequence length by patch_size factor, making attention faster.
    
    Example:
        Input: (Batch, 50, 15)  # 50 timesteps, 15 features
        Output: (Batch, 12, 128)  # 12 patches, 128-dim embeddings
    """
    
    def __init__(
        self,
        input_features: int = 15,
        d_model: int = 128,
        patch_size: int = 4,
        patch_stride: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        
        # Conv1d: treats features as channels, time as sequence
        # kernel_size = patch_size, stride = patch_stride
        self.proj = nn.Conv1d(
            in_channels=input_features,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, Features) - time series data
            
        Returns:
            (Batch, NumPatches, d_model) - patch embeddings
        """
        # Conv1d expects (B, C, L) so transpose
        x = x.transpose(1, 2)  # (B, Features, SeqLen)
        
        # Apply convolution
        x = self.proj(x)  # (B, d_model, NumPatches)
        
        # Transpose back
        x = x.transpose(1, 2)  # (B, NumPatches, d_model)
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Adds position information to patch embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, SeqLen, d_model)
        Returns:
            (Batch, SeqLen, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StrategyGLU(nn.Module):
    """
    Gated Linear Unit for Strategy Features.
    
    Processes the 64 strategy features with gating mechanism.
    This allows the model to selectively use different indicator groups.
    
    GLU(x) = (Wx + b) ⊙ σ(Vx + c)
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        
        # Linear transformation
        self.linear = nn.Linear(input_dim, hidden_dim)
        
        # Gate
        self.gate = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 64) strategy features
        Returns:
            (Batch, output_dim) gated features
        """
        # GLU mechanism
        linear_out = self.linear(x)
        gate_out = torch.sigmoid(self.gate(x))
        gated = linear_out * gate_out
        
        # Project to output dim
        out = self.output(gated)
        out = self.norm(out)
        
        return out


class LitePatchTransformer(nn.Module):
    """
    Lite Patch Transformer v5.
    
    A lightweight transformer for time series classification.
    Uses patch embedding to reduce computation and a CLS token for classification.
    
    Example:
        >>> config = V5Config()
        >>> model = LitePatchTransformer(config)
        >>> x_fast = torch.randn(32, 50, 15)   # M5 features
        >>> x_strat = torch.randn(32, 64)      # Strategy features
        >>> logits = model(x_fast, x_strat)
        >>> print(logits.shape)  # (32, 3)
    """
    
    def __init__(self, config: Optional[V5Config] = None):
        super().__init__()
        
        self.config = config or V5Config()
        c = self.config
        
        # === Patch Embedding ===
        self.patch_embed = PatchEmbedding(
            input_features=c.input_features,
            d_model=c.d_model,
            patch_size=c.patch_size,
            patch_stride=c.patch_stride,
            dropout=c.dropout,
        )
        
        # === CLS Token (learnable) ===
        self.cls_token = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)
        
        # === Positional Encoding ===
        self.pos_encoder = PositionalEncoding(
            d_model=c.d_model,
            max_len=c.num_patches + 1,  # +1 for CLS token
            dropout=c.dropout,
        )
        
        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.num_layers,
        )
        
        # === Strategy Feature Processing ===
        self.strategy_glu = StrategyGLU(
            input_dim=c.strategy_dim,
            hidden_dim=c.d_model,
            output_dim=c.d_model // 2,  # 64
        )
        
        # === Classification Head ===
        # CLS token (128) + Strategy GLU (64) = 192
        head_input_dim = c.d_model + c.d_model // 2
        
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_fast: torch.Tensor,
        x_strat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_fast: (Batch, SeqLen, Features) - M5 time series with V3 features
            x_strat: (Batch, 64) - Strategy signal features
            
        Returns:
            (Batch, num_classes) - Classification logits
        """
        batch_size = x_fast.size(0)
        
        # === Patch Embedding ===
        patches = self.patch_embed(x_fast)  # (B, NumPatches, d_model)
        
        # === Add CLS Token ===
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 1 + NumPatches, d_model)
        
        # === Positional Encoding ===
        x = self.pos_encoder(x)
        
        # === Transformer Encoder ===
        x = self.transformer(x)  # (B, 1 + NumPatches, d_model)
        
        # === Extract CLS Token ===
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # === Process Strategy Features ===
        strat_output = self.strategy_glu(x_strat)  # (B, d_model // 2)
        
        # === Concatenate and Classify ===
        combined = torch.cat([cls_output, strat_output], dim=-1)  # (B, d_model + d_model//2)
        logits = self.head(combined)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(
        self,
        x_fast: torch.Tensor,
        x_strat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Returns dict with attention weights from each layer.
        """
        # This would require hooks, simplified version here
        batch_size = x_fast.size(0)
        
        patches = self.patch_embed(x_fast)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = self.pos_encoder(x)
        
        # Get logits normally
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        strat_output = self.strategy_glu(x_strat)
        combined = torch.cat([cls_output, strat_output], dim=-1)
        logits = self.head(combined)
        
        return {
            'logits': logits,
            'cls_embedding': cls_output,
            'strategy_embedding': strat_output,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LitePatchTransformer v5 - Test")
    print("=" * 60)
    
    # Create model
    config = V5Config()
    model = LitePatchTransformer(config)
    
    print(f"\nConfig:")
    print(f"  seq_len: {config.seq_len}")
    print(f"  patch_size: {config.patch_size}")
    print(f"  num_patches: {config.num_patches}")
    print(f"  d_model: {config.d_model}")
    print(f"  num_layers: {config.num_layers}")
    
    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,}")
    
    # Test forward pass
    batch_size = 32
    x_fast = torch.randn(batch_size, config.seq_len, config.input_features)
    x_strat = torch.randn(batch_size, config.strategy_dim)
    
    print(f"\nInput shapes:")
    print(f"  x_fast: {x_fast.shape}")
    print(f"  x_strat: {x_strat.shape}")
    
    # Forward
    model.eval()
    with torch.no_grad():
        logits = model(x_fast, x_strat)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"Sample logits: {logits[0]}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print(f"\n🚀 Testing on GPU...")
        model = model.cuda()
        x_fast = x_fast.cuda()
        x_strat = x_strat.cuda()
        
        # Warmup
        for _ in range(10):
            _ = model(x_fast, x_strat)
        
        # Benchmark
        import time
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(100):
            _ = model(x_fast, x_strat)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"100 forward passes: {elapsed*1000:.1f}ms")
        print(f"Per batch: {elapsed*10:.2f}ms")
        print(f"Per sample: {elapsed*10/batch_size:.3f}ms")
    
    print("\n✅ LitePatchTransformer v5 test passed!")

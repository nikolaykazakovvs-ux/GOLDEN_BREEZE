"""
Golden Breeze Fusion Transformer v4.0 - Embeddings

Implements:
1. SlidingPatchEmbedding - Overlapping patch encoding for OHLCV sequences
2. SMCEmbedding - Smart Money Concepts feature encoding (static + dynamic)

Author: Golden Breeze Team
Version: 4.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SlidingPatchEmbedding(nn.Module):
    """
    Sliding Patch Embedding for OHLCV time series.
    
    Converts a sequence of OHLCV bars into overlapping patches,
    then projects each patch to d_model dimension.
    
    Architecture:
    - Input: (B, seq_len, 5) OHLCV data
    - Unfold into overlapping patches: (B, num_patches, patch_size * 5)
    - Linear projection: (B, num_patches, d_model)
    - Add positional encoding
    
    Example:
        >>> embed = SlidingPatchEmbedding(d_model=128, patch_size=16, patch_stride=8)
        >>> x = torch.randn(32, 200, 5)  # (batch, seq_len, OHLCV)
        >>> out = embed(x)  # (32, 24, 128) - 24 patches
    """
    
    def __init__(
        self,
        d_model: int = 128,
        patch_size: int = 16,
        patch_stride: int = 8,
        input_channels: int = 5,
        max_seq_len: int = 500,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Output embedding dimension
            patch_size: Size of each patch (number of bars)
            patch_stride: Stride between patches (overlap = patch_size - stride)
            input_channels: Number of input features per bar (5 for OHLCV)
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_channels = input_channels
        
        # Patch dimension after flattening
        self.patch_dim = patch_size * input_channels
        
        # Linear projection from patch to d_model
        self.projection = nn.Linear(self.patch_dim, d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding (learnable)
        max_patches = (max_seq_len - patch_size) // patch_stride + 1
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_patches, d_model) * 0.02
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # CLS token (optional, for pooling)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(
        self, 
        x: torch.Tensor,
        add_cls_token: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, seq_len, input_channels)
            add_cls_token: If True, prepend CLS token to sequence
            
        Returns:
            Embedded patches of shape (B, num_patches, d_model)
            or (B, num_patches + 1, d_model) if add_cls_token=True
        """
        B, seq_len, C = x.shape
        assert C == self.input_channels, \
            f"Expected {self.input_channels} channels, got {C}"
        
        # Transpose for unfold: (B, C, seq_len)
        x = x.transpose(1, 2)
        
        # Create overlapping patches using unfold
        # Output: (B, C, num_patches, patch_size)
        patches = x.unfold(
            dimension=2, 
            size=self.patch_size, 
            step=self.patch_stride
        )
        
        # Reshape: (B, num_patches, patch_size * C)
        num_patches = patches.shape[2]
        patches = patches.permute(0, 2, 3, 1).reshape(B, num_patches, -1)
        
        # Project to d_model
        x = self.projection(patches)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :num_patches, :]
        
        # Apply layer norm and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        # Optionally add CLS token
        if add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        return x
    
    def get_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for a given sequence length."""
        return (seq_len - self.patch_size) // self.patch_stride + 1


class SMCEmbedding(nn.Module):
    """
    Smart Money Concepts (SMC) Embedding.
    
    Encodes SMC features into two pathways:
    1. Static SMC → Bias vector (added to encoder hidden states)
    2. Dynamic SMC → Token sequence (prepended to input sequence)
    
    Static SMC Features:
    - Market structure (HH, HL, LH, LL counts)
    - Trend bias (bullish/bearish/neutral)
    - Liquidity zones (proximity to key levels)
    - Session context (Asian/London/NY)
    
    Dynamic SMC Features:
    - Order Blocks (OB)
    - Fair Value Gaps (FVG)
    - Breaker Blocks
    - Mitigation events
    
    Example:
        >>> smc_embed = SMCEmbedding(d_model=128, static_dim=16, dynamic_dim=16)
        >>> static_feats = torch.randn(32, 16)
        >>> dynamic_feats = torch.randn(32, 10, 16)  # 10 SMC events
        >>> bias, tokens = smc_embed(static_feats, dynamic_feats)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        static_dim: int = 16,
        dynamic_dim: int = 16,
        max_dynamic_tokens: int = 10,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Output embedding dimension
            static_dim: Dimension of static SMC features
            dynamic_dim: Dimension of each dynamic SMC event
            max_dynamic_tokens: Maximum number of dynamic tokens
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.max_dynamic_tokens = max_dynamic_tokens
        
        # Static pathway: features → bias vector
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Dynamic pathway: events → token embeddings
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        
        # Positional encoding for dynamic tokens
        self.dynamic_pos = nn.Parameter(
            torch.randn(1, max_dynamic_tokens, d_model) * 0.02
        )
        
        # Event type embedding (optional, for typed SMC events)
        self.event_type_embed = nn.Embedding(8, d_model)  # 8 event types
        
        # Special [SMC] token to mark SMC context
        self.smc_cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: Optional[torch.Tensor] = None,
        dynamic_mask: Optional[torch.Tensor] = None,
        event_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            static_features: Static SMC features (B, static_dim)
            dynamic_features: Dynamic SMC events (B, num_events, dynamic_dim)
            dynamic_mask: Mask for valid dynamic events (B, num_events)
            event_types: Event type indices (B, num_events) for type embedding
            
        Returns:
            Tuple of:
            - static_bias: Bias vector (B, d_model) to add to encoder states
            - dynamic_tokens: Token sequence (B, num_tokens, d_model)
        """
        B = static_features.shape[0]
        
        # === Static Pathway ===
        static_bias = self.static_encoder(static_features)  # (B, d_model)
        
        # === Dynamic Pathway ===
        if dynamic_features is None:
            # No dynamic events - return only SMC CLS token
            dynamic_tokens = self.smc_cls_token.expand(B, -1, -1)
        else:
            num_events = dynamic_features.shape[1]
            
            # Encode dynamic events
            dynamic_tokens = self.dynamic_encoder(dynamic_features)  # (B, N, d_model)
            
            # Add event type embedding if provided
            if event_types is not None:
                type_embed = self.event_type_embed(event_types)  # (B, N, d_model)
                dynamic_tokens = dynamic_tokens + type_embed
            
            # Add positional encoding
            dynamic_tokens = dynamic_tokens + self.dynamic_pos[:, :num_events, :]
            
            # Prepend SMC CLS token
            smc_cls = self.smc_cls_token.expand(B, -1, -1)
            dynamic_tokens = torch.cat([smc_cls, dynamic_tokens], dim=1)
            
            # Apply mask if provided (mask out padding)
            if dynamic_mask is not None:
                # Expand mask for CLS token
                cls_mask = torch.ones(B, 1, device=dynamic_mask.device, dtype=dynamic_mask.dtype)
                full_mask = torch.cat([cls_mask, dynamic_mask], dim=1)
                dynamic_tokens = dynamic_tokens * full_mask.unsqueeze(-1)
        
        return static_bias, dynamic_tokens
    
    def get_static_bias(self, static_features: torch.Tensor) -> torch.Tensor:
        """Get only the static bias vector."""
        return self.static_encoder(static_features)
    
    def get_dynamic_tokens(
        self, 
        dynamic_features: torch.Tensor
    ) -> torch.Tensor:
        """Get only the dynamic token sequence."""
        _, tokens = self.forward(
            static_features=torch.zeros(
                dynamic_features.shape[0], 
                self.static_dim,
                device=dynamic_features.device
            ),
            dynamic_features=dynamic_features
        )
        return tokens


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (for non-learnable variant).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

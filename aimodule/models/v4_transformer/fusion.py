"""
Golden Breeze Fusion Transformer v4.0 - Fusion Module

Implements Gated Cross-Attention for fusing Fast (M5) and Slow (H1) streams.

Fusion Equation:
    Attn = MultiHeadAttention(Q_fast, K_slow, V_slow)
    output = α * Attn + (1 - α) * Q_fast
    
where α is a learnable gating parameter (sigmoid activation).

Author: Golden Breeze Team
Version: 4.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention Fusion Layer.
    
    Combines Fast stream (Query) with Slow stream (Key, Value) using
    multi-head cross-attention, then applies learnable gating to control
    the fusion ratio.
    
    Architecture:
        1. Cross-attention: Fast queries attend to Slow keys/values
        2. Gating: α * attention_output + (1 - α) * fast_input
        3. Feed-forward network
        4. Layer normalization (pre-norm style)
    
    The gating parameter α allows the model to learn how much context
    from the slow stream is needed at each position.
    
    Example:
        >>> fusion = GatedCrossAttention(d_model=128, nhead=4)
        >>> fast = torch.randn(32, 24, 128)   # Fast stream patches
        >>> slow = torch.randn(32, 10, 128)   # Slow stream + SMC tokens
        >>> out = fusion(fast, slow)  # (32, 24, 128)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        gate_init: float = 0.5,
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout rate
            activation: Activation function ("gelu" or "relu")
            gate_init: Initial value for gating parameter (before sigmoid)
        """
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # === Cross-Attention ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # === Learnable Gating Parameter ===
        # Initialize so sigmoid(gate_init) ≈ 0.5 (balanced fusion)
        init_value = torch.tensor([gate_init]).log() - torch.tensor([1 - gate_init]).log()
        self.gate_param = nn.Parameter(init_value)
        
        # === Feed-Forward Network ===
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # === Layer Normalization (Pre-Norm) ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_slow = nn.LayerNorm(d_model)
        
        # === Dropout ===
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        slow_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            fast: Fast stream tensor (B, seq_fast, d_model) - Query
            slow: Slow stream tensor (B, seq_slow, d_model) - Key, Value
            slow_mask: Attention mask for slow stream (B, seq_slow)
            return_attention: If True, return attention weights
            
        Returns:
            Fused output (B, seq_fast, d_model)
            Optionally: attention weights (B, nhead, seq_fast, seq_slow)
        """
        # Pre-normalize
        fast_norm = self.norm1(fast)
        slow_norm = self.norm_slow(slow)
        
        # Cross-attention: Fast queries attend to Slow keys/values
        attn_output, attn_weights = self.cross_attn(
            query=fast_norm,
            key=slow_norm,
            value=slow_norm,
            key_padding_mask=slow_mask,
            need_weights=return_attention,
        )
        
        # Gating: α * attn + (1 - α) * fast
        alpha = torch.sigmoid(self.gate_param)
        gated_output = alpha * attn_output + (1 - alpha) * fast_norm
        
        # Residual connection
        x = fast + self.dropout(gated_output)
        
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x, None
    
    def get_gate_value(self) -> float:
        """Get current gating value (after sigmoid)."""
        return torch.sigmoid(self.gate_param).item()
    
    def set_gate_value(self, value: float):
        """Set gating value (will be converted through inverse sigmoid)."""
        assert 0 < value < 1, "Gate value must be in (0, 1)"
        inv_sigmoid = torch.tensor([value]).log() - torch.tensor([1 - value]).log()
        self.gate_param.data = inv_sigmoid


class StackedGatedCrossAttention(nn.Module):
    """
    Stacked Gated Cross-Attention layers for deeper fusion.
    
    Multiple GatedCrossAttention layers stacked with residual connections.
    Each layer can have different gating values, allowing hierarchical fusion.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of stacked fusion layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            GatedCrossAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                gate_init=0.5,
            )
            for _ in range(num_layers)
        ])
        
        self.num_layers = num_layers
    
    def forward(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        slow_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all fusion layers.
        
        Args:
            fast: Fast stream (B, seq_fast, d_model)
            slow: Slow stream (B, seq_slow, d_model)
            slow_mask: Mask for slow stream
            
        Returns:
            Fused output (B, seq_fast, d_model)
        """
        x = fast
        for layer in self.layers:
            x, _ = layer(x, slow, slow_mask)
        return x
    
    def get_gate_values(self) -> list:
        """Get gating values for all layers."""
        return [layer.get_gate_value() for layer in self.layers]


class AdditiveGatedFusion(nn.Module):
    """
    Simple Additive Gated Fusion (alternative to cross-attention).
    
    For cases where cross-attention is too expensive or when
    the slow stream is already aggregated to a single vector.
    
    Fusion:
        slow_pool = global_pool(slow)
        gate = sigmoid(linear(slow_pool))
        output = fast + gate * broadcast(slow_pool)
    """
    
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        fast: torch.Tensor, 
        slow: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fast: (B, seq_fast, d_model)
            slow: (B, seq_slow, d_model) or (B, d_model)
            
        Returns:
            Fused output (B, seq_fast, d_model)
        """
        # Pool slow stream if needed
        if slow.dim() == 3:
            slow_pooled = slow.mean(dim=1)  # (B, d_model)
        else:
            slow_pooled = slow
        
        # Compute gate
        gate = self.gate_proj(slow_pooled)  # (B, d_model)
        
        # Compute value to inject
        value = self.value_proj(slow_pooled)  # (B, d_model)
        
        # Broadcast and fuse
        gated_value = gate * value  # (B, d_model)
        gated_value = gated_value.unsqueeze(1)  # (B, 1, d_model)
        
        # Add to fast stream with residual
        output = fast + self.dropout(gated_value)
        output = self.norm(output)
        
        return output

"""
Golden Breeze Fusion Transformer v4.0 - Main Model

GoldenBreezeFusionV4: State-of-the-art Dual-Stream Transformer for
XAUUSD price direction prediction.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     GoldenBreezeFusionV4                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   [M5 OHLCV] ──► SlidingPatchEmbed ──► FastEncoder ──┐         │
    │                                                       │         │
    │                                      GatedCrossAttn ◄─┤         │
    │                                                       │         │
    │   [H1 OHLCV] ──► SlidingPatchEmbed ──► SlowEncoder ──┤         │
    │        +                                  + bias      │         │
    │   [SMC Static] ──► SMCEmbed.bias ─────────┘          │         │
    │   [SMC Dynamic] ──► SMCEmbed.tokens ──────┘          │         │
    │                                                       │         │
    │                                      Fusion Output ───┤         │
    │                                                       │         │
    │                           ┌── ScoreHead ──► [-1, 1]   │         │
    │                           │                           │         │
    │                           └── ClassHead ──► [UP/DOWN/HOLD]     │
    └─────────────────────────────────────────────────────────────────┘

Author: Golden Breeze Team
Version: 4.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import json
from pathlib import Path

from .config import V4Config
from .embeddings import SlidingPatchEmbedding, SMCEmbedding
from .fusion import GatedCrossAttention, StackedGatedCrossAttention


class GoldenBreezeFusionV4(nn.Module):
    """
    Golden Breeze Fusion Transformer v4.0
    
    Dual-stream architecture that processes:
    - Fast stream (M5): High-frequency price action with sliding patches
    - Slow stream (H1): Lower-frequency context with SMC features
    
    Outputs:
    - Classification: UP / DOWN / HOLD logits
    - Continuous score: [-1, 1] for risk management
    
    Example:
        >>> config = V4Config()
        >>> model = GoldenBreezeFusionV4(config)
        >>> 
        >>> # Dummy inputs
        >>> x_fast = torch.randn(32, 200, 5)    # M5 OHLCV
        >>> x_slow = torch.randn(32, 50, 5)     # H1 OHLCV
        >>> smc_static = torch.randn(32, 16)    # Static SMC features
        >>> smc_dynamic = torch.randn(32, 10, 16)  # Dynamic SMC events
        >>> 
        >>> out = model(x_fast, x_slow, smc_static, smc_dynamic)
        >>> print(out["class_logits"].shape)  # (32, 3)
        >>> print(out["score"].shape)         # (32, 1)
    """
    
    def __init__(self, config: Optional[V4Config] = None):
        """
        Initialize the model.
        
        Args:
            config: V4Config instance. If None, uses default config.
        """
        super().__init__()
        
        self.config = config or V4Config()
        c = self.config
        
        # === Embeddings ===
        self.fast_embed = SlidingPatchEmbedding(
            d_model=c.d_model,
            patch_size=c.patch_size,
            patch_stride=c.patch_stride,
            input_channels=c.input_channels,
            max_seq_len=c.seq_len_fast * 2,  # Buffer for longer sequences
            dropout=c.dropout,
        )
        
        self.slow_embed = SlidingPatchEmbedding(
            d_model=c.d_model,
            patch_size=c.patch_size,
            patch_stride=c.patch_stride,
            input_channels=c.input_channels,
            max_seq_len=c.seq_len_slow * 2,
            dropout=c.dropout,
        )
        
        self.smc_embed = SMCEmbedding(
            d_model=c.d_model,
            static_dim=c.static_smc_dim,
            dynamic_dim=c.dynamic_smc_dim,
            max_dynamic_tokens=c.max_dynamic_tokens,
            dropout=c.dropout,
        )
        
        # === Fast Encoder (M5 Stream) ===
        fast_encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation=c.activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.fast_encoder = nn.TransformerEncoder(
            fast_encoder_layer,
            num_layers=c.num_layers_fast,
        )
        
        # === Slow Encoder (H1 Stream) ===
        slow_encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation=c.activation,
            batch_first=True,
            norm_first=True,
        )
        self.slow_encoder = nn.TransformerEncoder(
            slow_encoder_layer,
            num_layers=c.num_layers_slow,
        )
        
        # === Fusion Layer ===
        self.fusion = GatedCrossAttention(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            activation=c.activation,
        )
        
        # === Output Heads ===
        
        # Classification Head: UP / DOWN / HOLD
        self.class_head = nn.Sequential(
            nn.Linear(c.d_model, c.score_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.score_hidden_dim, c.num_classes),
        )
        
        # Continuous Score Head: [-1, 1]
        self.score_head = nn.Sequential(
            nn.Linear(c.d_model, c.score_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.score_hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # === Pooling ===
        self.pool_type = "mean"  # Can be "mean", "cls", or "attention"
        
        # Attention pooling (optional)
        self.attn_pool = nn.Sequential(
            nn.Linear(c.d_model, 1),
            nn.Softmax(dim=1),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_fast_ohlcv: torch.Tensor,
        x_slow_ohlcv: torch.Tensor,
        smc_static: torch.Tensor,
        smc_dynamic: Optional[torch.Tensor] = None,
        smc_dynamic_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_fast_ohlcv: Fast stream M5 OHLCV (B, seq_fast, 5)
            x_slow_ohlcv: Slow stream H1 OHLCV (B, seq_slow, 5)
            smc_static: Static SMC features (B, static_smc_dim)
            smc_dynamic: Dynamic SMC events (B, num_events, dynamic_smc_dim)
            smc_dynamic_mask: Mask for valid dynamic events (B, num_events)
            return_features: If True, also return intermediate features
            
        Returns:
            Dict with:
            - "class_logits": (B, num_classes) classification logits
            - "class_probs": (B, num_classes) softmax probabilities
            - "score": (B, 1) continuous score in [-1, 1]
            - "predicted_class": (B,) predicted class indices
            - "gate_value": float, current gating value
            - "features": (B, d_model) pooled features (if return_features=True)
        """
        B = x_fast_ohlcv.shape[0]
        
        # === Embedding ===
        
        # Fast stream: M5 patches
        fast_patches = self.fast_embed(x_fast_ohlcv)  # (B, num_patches_fast, d_model)
        
        # Slow stream: H1 patches
        slow_patches = self.slow_embed(x_slow_ohlcv)  # (B, num_patches_slow, d_model)
        
        # SMC embedding
        smc_bias, smc_tokens = self.smc_embed(
            static_features=smc_static,
            dynamic_features=smc_dynamic,
            dynamic_mask=smc_dynamic_mask,
        )
        
        # === Fast Encoder ===
        fast_encoded = self.fast_encoder(fast_patches)  # (B, num_patches_fast, d_model)
        
        # === Slow Encoder with SMC Bias ===
        # Add static SMC bias to slow patches
        slow_patches_biased = slow_patches + smc_bias.unsqueeze(1)
        
        # Prepend dynamic SMC tokens
        slow_with_smc = torch.cat([smc_tokens, slow_patches_biased], dim=1)
        
        # Encode slow stream
        slow_encoded = self.slow_encoder(slow_with_smc)  # (B, num_smc + num_patches_slow, d_model)
        
        # === Fusion ===
        fused, _ = self.fusion(
            fast=fast_encoded,
            slow=slow_encoded,
        )  # (B, num_patches_fast, d_model)
        
        # === Pooling ===
        if self.pool_type == "mean":
            pooled = fused.mean(dim=1)  # (B, d_model)
        elif self.pool_type == "attention":
            attn_weights = self.attn_pool(fused)  # (B, num_patches, 1)
            pooled = (fused * attn_weights).sum(dim=1)  # (B, d_model)
        else:  # cls - use first token
            pooled = fused[:, 0, :]  # (B, d_model)
        
        # === Output Heads ===
        class_logits = self.class_head(pooled)  # (B, num_classes)
        class_probs = F.softmax(class_logits, dim=-1)
        predicted_class = class_logits.argmax(dim=-1)
        
        score = self.score_head(pooled)  # (B, 1)
        
        # Build output dict
        output = {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "score": score,
            "predicted_class": predicted_class,
            "gate_value": self.fusion.get_gate_value(),
        }
        
        if return_features:
            output["features"] = pooled
            output["fast_encoded"] = fast_encoded
            output["slow_encoded"] = slow_encoded
            output["fused"] = fused
        
        return output
    
    def predict(
        self,
        x_fast_ohlcv: torch.Tensor,
        x_slow_ohlcv: torch.Tensor,
        smc_static: torch.Tensor,
        smc_dynamic: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Simplified prediction method for inference.
        
        Returns:
            Dict with prediction results:
            - "direction": "UP" / "DOWN" / "HOLD"
            - "confidence": float in [0, 1]
            - "score": float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x_fast_ohlcv, x_slow_ohlcv, smc_static, smc_dynamic)
        
        class_names = ["DOWN", "HOLD", "UP"]  # Assuming class order
        pred_idx = out["predicted_class"][0].item()
        confidence = out["class_probs"][0, pred_idx].item()
        score = out["score"][0, 0].item()
        
        return {
            "direction": class_names[pred_idx],
            "confidence": confidence,
            "score": score,
            "class_probs": {
                name: out["class_probs"][0, i].item() 
                for i, name in enumerate(class_names)
            },
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            "fast_embed": count(self.fast_embed),
            "slow_embed": count(self.slow_embed),
            "smc_embed": count(self.smc_embed),
            "fast_encoder": count(self.fast_encoder),
            "slow_encoder": count(self.slow_encoder),
            "fusion": count(self.fusion),
            "class_head": count(self.class_head),
            "score_head": count(self.score_head),
            "total": count(self),
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "version": "4.0.0",
        }
        torch.save(checkpoint, path)
        
        # Also save config as JSON
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "GoldenBreezeFusionV4":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        config = V4Config.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        return model


def create_dummy_inputs(config: V4Config, batch_size: int = 2, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Create dummy inputs for testing.
    
    Returns dict with all required inputs for forward pass.
    """
    return {
        "x_fast_ohlcv": torch.randn(batch_size, config.seq_len_fast, config.input_channels, device=device),
        "x_slow_ohlcv": torch.randn(batch_size, config.seq_len_slow, config.input_channels, device=device),
        "smc_static": torch.randn(batch_size, config.static_smc_dim, device=device),
        "smc_dynamic": torch.randn(batch_size, config.max_dynamic_tokens, config.dynamic_smc_dim, device=device),
    }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("GoldenBreezeFusionV4 - Quick Test")
    print("=" * 60)
    
    # Create model
    config = V4Config()
    model = GoldenBreezeFusionV4(config)
    
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  nhead: {config.nhead}")
    print(f"  num_layers_fast: {config.num_layers_fast}")
    print(f"  num_layers_slow: {config.num_layers_slow}")
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nParameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    inputs = create_dummy_inputs(config, batch_size=4)
    
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    
    print(f"\nOutputs:")
    print(f"  class_logits: {output['class_logits'].shape}")
    print(f"  class_probs: {output['class_probs'].shape}")
    print(f"  score: {output['score'].shape}")
    print(f"  gate_value: {output['gate_value']:.4f}")
    
    # Test prediction
    print(f"\nTesting prediction...")
    pred = model.predict(**{k: v[:1] for k, v in inputs.items()})
    print(f"  direction: {pred['direction']}")
    print(f"  confidence: {pred['confidence']:.4f}")
    print(f"  score: {pred['score']:.4f}")
    
    print("\n✅ All tests passed!")

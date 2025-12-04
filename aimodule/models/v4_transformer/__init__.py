"""
Golden Breeze Fusion Transformer v4.0

State-of-the-art Dual-Stream Sliding-Patch Transformer architecture
for XAUUSD price direction prediction.

Architecture:
- Fast Stream (M5): Sliding Patch Encoding for micro-structure
- Slow Stream (H1): SMC context with static bias + dynamic tokens  
- Gated Cross-Attention Fusion
- Dual-Head Output (classification + continuous score)

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

from .config import V4Config
from .embeddings import SlidingPatchEmbedding, SMCEmbedding
from .fusion import GatedCrossAttention
from .model import GoldenBreezeFusionV4

__all__ = [
    "V4Config",
    "SlidingPatchEmbedding", 
    "SMCEmbedding",
    "GatedCrossAttention",
    "GoldenBreezeFusionV4",
]

__version__ = "4.0.0"

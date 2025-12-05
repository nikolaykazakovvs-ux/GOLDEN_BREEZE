"""
Golden Breeze v5 Ultimate - Multi-Timeframe Transformer with Gated Feature Mixer

Архитектура:
- 4 виртуальных таймфрейма: M5, M15 (виртуальный), H1, H4 (виртуальный)
- Gated Feature Mixer для разрешения конфликтов между индикаторами
- Lite Patch Transformer для эффективной обработки
- Hardware optimized для RTX 3070 + Ryzen 7

Author: Golden Breeze Team
Version: 5.1.0 Ultimate
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass


@dataclass
class V5UltimateConfig:
    """Конфигурация v5 Ultimate."""
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.3
    
    # Patch sizes
    patch_m5: int = 4       # 50 bars -> 12 patches
    patch_h1: int = 2       # 20 bars -> 10 patches
    
    # Input dims
    fast_features: int = 15
    slow_features: int = 8
    strategy_features: int = 64
    
    num_classes: int = 3


class PatchEmbedding(nn.Module):
    """
    Превращает кусок графика в сжатые токены (Patching).
    Используется для M5, M15, H1, H4 потоков.
    """
    def __init__(self, input_dim: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(input_dim, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Seq, Feat) -> (B, Feat, Seq)
        x = x.transpose(1, 2)
        x = self.proj(x)            # (B, d_model, patches)
        x = x.transpose(1, 2)       # (B, patches, d_model)
        return self.norm(x)


class GatedFeatureMixer(nn.Module):
    """
    РЕШЕНИЕ КОНФЛИКТОВ:
    Разделяет 64 индикатора на группы и взвешивает их через Gating Network.
    
    Группы (на основе анализа конфликтов):
    - Trend: EMA, SMA, MACD - тренд-следящие
    - Oscillator: RSI, Stochastic, Williams - осцилляторы (КОНФЛИКТ с Trend!)
    - Volatility: ATR, Bollinger - волатильность
    - Volume/Pattern: Volume, Candle patterns - объём и паттерны
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        
        # Разделение на 4 группы по 16 фич
        # (в реальности границы приблизительные, основаны на анализе)
        self.groups = {
            'trend': slice(0, 16),       # Тренд-индикаторы
            'oscillator': slice(16, 32), # Осцилляторы (RSI, Stoch, Williams)
            'volatility': slice(32, 48), # Волатильность (ATR, BB, др)
            'pattern': slice(48, 64)     # Паттерны и объём
        }
        
        # Для каждой группы: Gate (вес важности) + Projection
        self.gates = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        
        for name, sl in self.groups.items():
            dim = sl.stop - sl.start  # 16
            
            # Gate: решает насколько важна группа (0..1)
            self.gates[name] = nn.Sequential(
                nn.Linear(dim, 8),
                nn.GELU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
            
            # Projection: преобразует группу в d_model
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, d_model),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 64) strategy features
        Returns:
            (Batch, d_model) gated mixed context
        """
        mixed_context = torch.zeros(x.size(0), self.projections['trend'][0].out_features, 
                                     device=x.device, dtype=x.dtype)
        
        gate_weights = {}
        
        for name, sl in self.groups.items():
            group_data = x[:, sl]                         # (B, 16)
            weight = self.gates[name](group_data)         # (B, 1)
            feat = self.projections[name](group_data)     # (B, d_model)
            mixed_context = mixed_context + feat * weight # Взвешенная сумма
            gate_weights[name] = weight.mean().item()     # Для отладки
        
        return self.final_norm(self.output_proj(mixed_context))


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for variable-length sequences."""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class GoldenBreezeV5Ultimate(nn.Module):
    """
    Golden Breeze v5 Ultimate - Multi-Timeframe Transformer
    
    Inputs:
        x_fast: (B, 50, 15) - M5 OHLCV + indicators
        x_slow: (B, 20, 8)  - H1 OHLCV + indicators  
        x_strat: (B, 64)    - Strategy indicators
        
    Output:
        (B, 3) - logits for [DOWN, NEUTRAL, UP]
    """
    
    def __init__(self, config: V5UltimateConfig = None):
        super().__init__()
        self.config = config or V5UltimateConfig()
        cfg = self.config
        
        # --- 1. Multi-Timeframe Patch Embeddings ---
        # M5 Stream: 50 bars, patch_size=4 -> 12 patches
        self.patch_m5 = PatchEmbedding(cfg.fast_features, cfg.d_model, cfg.patch_m5)
        
        # H1 Stream: 20 bars, patch_size=2 -> 10 patches
        self.patch_h1 = PatchEmbedding(cfg.slow_features, cfg.d_model, cfg.patch_h1)
        
        # --- 2. Gated Strategy Context (Conflict Resolution) ---
        self.mixer = GatedFeatureMixer(cfg.strategy_features, cfg.d_model)
        
        # --- 3. Positional Encoding ---
        self.pos_encoder = LearnablePositionalEncoding(cfg.d_model, max_len=200)
        
        # --- 4. Transformer Core ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        
        # --- 5. Classification Head ---
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for transformer stability."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(self, x_fast: torch.Tensor, x_slow: torch.Tensor, x_strat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with virtual multi-timeframe.
        
        Args:
            x_fast: (B, 50, 15) M5 bars
            x_slow: (B, 20, 8) H1 bars
            x_strat: (B, 64) Strategy features
            
        Returns:
            (B, 3) class logits
        """
        # === A. Виртуальный Multi-Timeframe ===
        
        # 1. M5 Stream: 50 bars -> 12 patches
        tokens_m5 = self.patch_m5(x_fast)  # (B, 12, d_model)
        
        # 2. M15 Stream (Virtual): slice M5 every 3 -> ~16 bars -> 4 patches
        x_m15 = x_fast[:, ::3, :]  # (B, 16, 15)
        tokens_m15 = self.patch_m5(x_m15)  # (B, 4, d_model)
        
        # 3. H1 Stream: 20 bars -> 10 patches
        tokens_h1 = self.patch_h1(x_slow)  # (B, 10, d_model)
        
        # 4. H4 Stream (Virtual): slice H1 every 4 -> 5 bars -> 2 patches
        x_h4 = x_slow[:, ::4, :]  # (B, 5, 8)
        tokens_h4 = self.patch_h1(x_h4)  # (B, 2, d_model)
        
        # === B. Strategy Context (Gated) ===
        # Получаем "Strategy Token", который разрешил конфликты индикаторов
        strat_token = self.mixer(x_strat).unsqueeze(1)  # (B, 1, d_model)
        
        # === C. Fusion ===
        # Собираем всё в одну последовательность:
        # [Strategy_Token, M5_patches..., M15_patches..., H1_patches..., H4_patches...]
        seq = torch.cat([strat_token, tokens_m5, tokens_m15, tokens_h1, tokens_h4], dim=1)
        # seq shape: (B, 1+12+4+10+2, d_model) = (B, 29, 128)
        
        seq = self.pos_encoder(seq)
        
        # === D. Transformer ===
        out = self.transformer(seq)  # (B, 29, d_model)
        
        # === E. Classify ===
        # Используем Strategy Token (индекс 0) как агрегатор контекста
        cls_repr = out[:, 0, :]  # (B, d_model)
        
        return self.head(cls_repr)  # (B, 3)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Для обратной совместимости
V5Config = V5UltimateConfig


if __name__ == "__main__":
    # Quick test
    config = V5UltimateConfig()
    model = GoldenBreezeV5Ultimate(config)
    
    print(f"Model: GoldenBreezeV5Ultimate")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward
    B = 4
    x_fast = torch.randn(B, 50, 15)
    x_slow = torch.randn(B, 20, 8)
    x_strat = torch.randn(B, 64)
    
    out = model(x_fast, x_slow, x_strat)
    print(f"Output shape: {out.shape}")  # (4, 3)
    print("✅ Model test passed!")

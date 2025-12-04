# Golden Breeze — PAT Architecture v4.0

**Version:** 4.0.0  
**Date:** 2025-12-04  
**Status:** IMPLEMENTED (Skeleton)  
**Branch:** `fusion-transformer-v4`

---

## 📋 Overview

Golden Breeze v4.0 использует **Gated Dual-Stream Sliding-Patch Transformer** — архитектуру нового поколения для предсказания направления XAUUSD.

### Ключевые инновации:
- 🔹 **Sliding Patch Encoding** — overlapping patches вместо token-per-bar
- 🔹 **Dual-Stream Processing** — Fast (M5) + Slow (H1) потоки
- 🔹 **SMC Embedding Injection** — Smart Money Concepts в архитектуру
- 🔹 **Gated Cross-Attention Fusion** — обучаемое слияние потоков
- 🔹 **Dual-Head Output** — классификация + непрерывный score

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GoldenBreezeFusionV4                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐                                                      │
│   │  M5 OHLCV    │──► SlidingPatchEmbed ──► FastEncoder (2L) ──┐       │
│   │  (200 bars)  │         ↓                                    │       │
│   └──────────────┘    [24 patches]                              │       │
│                                                      GatedCrossAttn     │
│   ┌──────────────┐                                              │       │
│   │  H1 OHLCV    │──► SlidingPatchEmbed ──► SlowEncoder (2L) ──┤       │
│   │  (50 bars)   │         ↓                  + SMC Bias        │       │
│   └──────────────┘    [5 patches]                               │       │
│                                                                  │       │
│   ┌──────────────┐    ┌──────────────┐                          │       │
│   │  SMC Static  │──► │  SMCEmbed    │──► Bias Vector ──────────┤       │
│   │  (16 dim)    │    │              │                          │       │
│   └──────────────┘    │              │                          │       │
│   ┌──────────────┐    │              │                          │       │
│   │  SMC Dynamic │──► │              │──► Dynamic Tokens ───────┘       │
│   │  (10×16)     │    └──────────────┘                                  │
│   └──────────────┘                                                      │
│                                                                         │
│                              ┌──────────────┐                           │
│                              │ Fusion Output│                           │
│                              │  (24, 128)   │                           │
│                              └──────┬───────┘                           │
│                                     │                                   │
│                              ┌──────┴───────┐                           │
│                              │   Pooling    │                           │
│                              │   (mean)     │                           │
│                              └──────┬───────┘                           │
│                                     │                                   │
│                     ┌───────────────┼───────────────┐                   │
│                     ▼                               ▼                   │
│              ┌─────────────┐                 ┌─────────────┐            │
│              │  ClassHead  │                 │  ScoreHead  │            │
│              │  MLP → 3    │                 │  MLP → 1    │            │
│              └──────┬──────┘                 └──────┬──────┘            │
│                     ▼                               ▼                   │
│              [DOWN/HOLD/UP]                   [-1.0, 1.0]               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
aimodule/models/v4_transformer/
├── __init__.py         # Module exports
├── config.py           # V4Config dataclass
├── embeddings.py       # SlidingPatchEmbedding, SMCEmbedding
├── fusion.py           # GatedCrossAttention, StackedGatedCrossAttention
└── model.py            # GoldenBreezeFusionV4 main class
```

---

## 🔧 Configuration (V4Config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Transformer hidden dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers_fast` | 2 | Fast encoder layers |
| `num_layers_slow` | 2 | Slow encoder layers |
| `dim_feedforward` | 512 | FFN hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `patch_size` | 16 | Size of sliding window patch |
| `patch_stride` | 8 | Stride between patches |
| `static_smc_dim` | 16 | Static SMC feature dimension |
| `dynamic_smc_dim` | 16 | Dynamic SMC event dimension |
| `seq_len_fast` | 200 | M5 sequence length |
| `seq_len_slow` | 50 | H1 sequence length |
| `num_classes` | 3 | Output classes (UP/DOWN/HOLD) |

---

## 📊 Components

### 1. SlidingPatchEmbedding

Converts OHLCV sequence into overlapping patches:

```python
# Input: (B, 200, 5) - 200 M5 bars with OHLCV
# Patch size: 16, Stride: 8
# Output: (B, 24, 128) - 24 overlapping patches

patches = SlidingPatchEmbedding(d_model=128, patch_size=16, patch_stride=8)
out = patches(x)  # Includes positional encoding
```

**Key features:**
- Uses `torch.nn.Unfold` for efficient sliding window
- Learnable positional encoding
- Optional CLS token

### 2. SMCEmbedding

Encodes Smart Money Concepts into two pathways:

```python
smc = SMCEmbedding(d_model=128, static_dim=16, dynamic_dim=16)

# Static pathway: Market structure bias
static_bias = smc.get_static_bias(static_features)  # (B, 128)

# Dynamic pathway: OB/FVG/BB events as tokens
_, dynamic_tokens = smc(static_features, dynamic_events)  # (B, 11, 128)
```

**Static SMC features (16 dim):**
- HH/HL/LH/LL counts
- Trend bias score
- Liquidity zone proximity
- Session context (Asian/London/NY)

**Dynamic SMC events (10 × 16 dim):**
- Order Blocks (bullish/bearish)
- Fair Value Gaps
- Breaker Blocks
- Mitigation events

### 3. GatedCrossAttention

Fuses Fast and Slow streams with learnable gating:

```python
fusion = GatedCrossAttention(d_model=128, nhead=4)

# Fast queries attend to Slow keys/values
# α = sigmoid(learnable_param)
# output = α * Attn(Q_fast, K_slow, V_slow) + (1-α) * Q_fast

fused, _ = fusion(fast_encoded, slow_encoded)
print(f"Gate α = {fusion.get_gate_value():.4f}")  # e.g., 0.6234
```

**Gating mechanism:**
- α starts at 0.5 (balanced)
- Learns optimal fusion ratio during training
- Allows model to decide how much context is needed

### 4. GoldenBreezeFusionV4

Main model class:

```python
config = V4Config()
model = GoldenBreezeFusionV4(config)

output = model(
    x_fast_ohlcv=m5_data,      # (B, 200, 5)
    x_slow_ohlcv=h1_data,      # (B, 50, 5)
    smc_static=smc_static,     # (B, 16)
    smc_dynamic=smc_dynamic,   # (B, 10, 16)
)

# Outputs:
# - class_logits: (B, 3)
# - class_probs: (B, 3) 
# - score: (B, 1) in [-1, 1]
# - predicted_class: (B,)
# - gate_value: float
```

---

## 📈 Parameter Count

| Component | Parameters |
|-----------|------------|
| fast_embed | ~20K |
| slow_embed | ~20K |
| smc_embed | ~35K |
| fast_encoder | ~265K |
| slow_encoder | ~265K |
| fusion | ~135K |
| class_head | ~8.5K |
| score_head | ~8.5K |
| **Total** | **~750K** |

*~13x больше параметров чем v3 LSTM (58K), но всё ещё легковесная модель.*

---

## 🧪 Quick Test

```python
from aimodule.models.v4_transformer import GoldenBreezeFusionV4, V4Config

# Create model
config = V4Config()
model = GoldenBreezeFusionV4(config)
model.eval()

# Dummy inputs
import torch
inputs = {
    "x_fast_ohlcv": torch.randn(1, 200, 5),
    "x_slow_ohlcv": torch.randn(1, 50, 5),
    "smc_static": torch.randn(1, 16),
    "smc_dynamic": torch.randn(1, 10, 16),
}

# Forward pass
with torch.no_grad():
    out = model(**inputs)

print(f"Prediction: {['DOWN', 'HOLD', 'UP'][out['predicted_class'][0]]}")
print(f"Confidence: {out['class_probs'].max():.2%}")
print(f"Score: {out['score'][0, 0]:.4f}")
```

---

## 🚀 Next Steps (Phase 2: Training)

1. **Data Pipeline**
   - Prepare M5 + H1 aligned datasets
   - Calculate SMC features (static + dynamic)
   - Create train/val/test splits

2. **Training Loop**
   - Multi-task loss: CrossEntropy + MSE
   - AdamW optimizer with cosine scheduler
   - Gradient clipping + label smoothing

3. **Evaluation**
   - Classification metrics: MCC, F1, Accuracy
   - Score metrics: MAE, correlation
   - Backtest integration

4. **Deployment**
   - ONNX export for inference
   - Integration with existing AIClient
   - A/B testing vs v3 LSTM

---

## ⚠️ Important Notes

1. **v3 Compatibility**: This architecture lives in a separate directory and does NOT modify v3 code.

2. **Git Workflow**: 
   - Stable v3: `core-v3-stable` branch
   - v4 development: `fusion-transformer-v4` branch

3. **No Training Yet**: This is skeleton code only. Training will be implemented in Phase 2.

---

*Documentation generated: 2025-12-04*

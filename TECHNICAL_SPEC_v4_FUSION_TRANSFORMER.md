# Golden Breeze — Technical Specification for Fusion Transformer v4.0

**Date:** 2025-12-04  
**Author:** Светомир + Умник  
**Status:** APPROVED & IMPLEMENTED  
**Branch:** `fusion-transformer-v4`  
**Stable baseline:** `core-v3-stable`

---

# 1. Overview

Golden Breeze v4.0 — это новая архитектура AI Core, основанная на:
- Sliding Patch Encoding  
- Dual-Stream Transformer  
- Cross-Timeframe Fusion  
- SMC Embedding Injection  
- Gated Cross-Attention  
- Dual-Head Output (classification + continuous score)

Цель: заменить устаревшую LSTM и обеспечить SOTA-уровень обработки ценовых данных.

---

# 2. Git Workflow

## New branch structure:

### 1) `core-v3-stable`  
Зафиксированная, неизменяемая ветка с работающим v3 AI Core.

### 2) `fusion-transformer-v4`  
Рабочая ветка для новой архитектуры.

### 3) Workflow rules
- v3 код **не изменяется**.  
- Все файлы v4 хранятся только в `aimodule/models/v4_transformer/`.  
- Ветка v4 не должна нарушать работу v3 inference.  
- Обязательный PR review перед merge в main.

---

# 3. Directory Structure

```
aimodule/models/v4_transformer/
├── __init__.py         ✅ IMPLEMENTED
├── config.py           ✅ IMPLEMENTED
├── embeddings.py       ✅ IMPLEMENTED
├── fusion.py           ✅ IMPLEMENTED
└── model.py            ✅ IMPLEMENTED
```

---

# 4. Architecture Specification (v4.0)

## 4.1 Inputs

### Stream A (Fast):
- M5 OHLCV (200 bars)
- Sliding Patch Encoding  
- Captures micro-structure and execution signals

### Stream B (Slow):
- H1 OHLCV (50 bars)
- SMC static features (bias)  
- SMC dynamic tokens (event embeddings)

---

## 4.2 Encoding Layers

### SlidingPatchEmbedding
- Uses `nn.Unfold`
- Patch size: 16  
- Stride: 8  
- Projects each patch → `d_model=128`
- Learnable positional encoding

### SMCEmbedding
Two branches:
1. **Static SMC → bias** (added to Slow encoder)
2. **Dynamic SMC → tokens** (prepended to sequence)

---

## 4.3 Dual-Stream Encoders

### Fast Encoder (M5)
- 2 layers  
- `d_model=128`, `nhead=4`
- Pre-norm TransformerEncoder

### Slow Encoder (H1 + SMC)
- 2 layers  
- Includes static SMC bias injection
- Dynamic SMC tokens prepended

---

## 4.4 Fusion: Gated Cross-Attention

Fusion equation:

```
Attn = MultiHeadAttention(Q_fast, K_slow, V_slow)
output = α * Attn + (1 - α) * Q_fast
```

α — learnable parameter (sigmoid), initialized at 0.5.

---

## 4.5 Outputs

### 1) Classification Head
```
UP / DOWN / HOLD → 3 logits
```

### 2) Continuous Score Head  
```
MLP → Tanh → [-1, 1]
```
Используется для риск-менеджмента и confidence calibration.

---

# 5. Acceptance Criteria — STATUS

| AC | Description | Status |
|----|-------------|--------|
| AC-1 | Model must compile and forward pass | ✅ PASSED |
| AC-2 | No impact on v3 code | ✅ PASSED |
| AC-3 | Documentation generated | ✅ PASSED |
| AC-4 | Unit tests structure | ⏳ TODO (Phase 2) |

---

# 6. Dependencies

- PyTorch ≥ 2.2  ✅
- CUDA ≥ 12.1    ✅ (RTX 3070 detected)
- Python ≥ 3.10  ✅

---

# 7. Deliverables — STATUS

| Deliverable | Status |
|-------------|--------|
| Skeleton code structure | ✅ COMPLETE |
| Documentation (PAT Architecture v4) | ✅ COMPLETE |
| Unit tests | ⏳ Phase 2 |
| Training pipeline | ⏳ Phase 2 |
| Integration with AIClient | ⏳ Phase 3 |

---

# 8. Parameter Summary

| Component | Parameters |
|-----------|------------|
| Total | ~750K |
| vs v3 LSTM | 13x larger |
| Memory footprint | ~3 MB |

---

# END OF SPEC

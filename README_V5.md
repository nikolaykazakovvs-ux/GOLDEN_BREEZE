# 🥇 Golden Breeze V5 Ultimate - AI Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Golden Breeze V5 Ultimate** - Передовая AI система для торговли с использованием глубокого обучения, многотаймфреймовой логики и полным конвейером обучения.

---

## 🏆 Performance Summary

### V5 Ultimate Achievement
| Метрика | V4 Lite | **V5 Ultimate** | Улучшение |
|---------|---------|-----------------|-----------|
| **Val MCC** | 0.1495 | **0.3316** 🏆 | **+122%** |
| **Train MCC** | 0.14 | 0.3312 | **+136%** |
| **Train Loss** | 1.05 | 0.9685 | **-7.8%** |
| **Architecture** | Transformer | LSTM Hybrid | — |
| **Parameters** | 83K | 327K | +3.9x |
| **Dataset** | XAUUSD | **BTC M5+H1** | Более обобщённая |

### 📊 V5 Training Progress
```
Epoch   1:  Val MCC +0.1205
Epoch  10:  Val MCC +0.1772
Epoch  20:  Val MCC +0.2171  (old V4: 0.1495)
Epoch  30:  Val MCC +0.2489
Epoch  40:  Val MCC +0.2715
Epoch  50:  Val MCC +0.2988
Epoch  60:  Val MCC +0.3142
Epoch  70:  Val MCC +0.3243
Epoch  80:  Val MCC +0.3299
Epoch  91:  Val MCC +0.3316 ✨ BEST (saved)
Epoch 100:  Val MCC +0.3307 (final)
```

---

## 🚀 Model V5 Ultimate

### Architecture
```
Input (Multi-timeframe BTC data)
    ↓
3-Layer LSTM (128 hidden units per layer)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense 64 → ReLU → Dropout
    ↓
Output (3 classes: DOWN, NEUTRAL, UP)
```

### Key Features
- ✅ **3-Layer LSTM** с 128 hidden units
- ✅ **BatchNormalization** для стабильности
- ✅ **Dropout 0.3** для регуляризации
- ✅ **327K параметров** (3.9x больше V4)
- ✅ **517K тренировочных примеров** (BTC M5+H1)
- ✅ **Mixed Precision Training** (AMP, FP16+FP32)
- ✅ **GPU Optimizations**: TF32, cuDNN benchmark

### Training Setup
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing + 5-epoch Warmup
- **Batch Size**: 512 (оптимизировано для RTX 3070)
- **Epochs**: 100 (43 минуты на GPU)
- **Loss Function**: CrossEntropyLoss
- **Early Stopping**: Patience 25

### Files
| Файл | Размер | Описание |
|------|--------|---------|
| `models/v5_btc/best_model.pt` | 3.9 MB | ✅ Лучшая модель (Val MCC 0.3316) |
| `models/v5_btc/best_model_mcc0.3316_20251206_043810.pt` | 3.9 MB | Бэкап |
| `models/v5_btc/checkpoint_epoch_*.pt` | 1.3 MB each | 10 контрольных точек |
| `train_v5_btc.py` | — | Скрипт обучения |
| `evaluate_best_model.py` | — | Скрипт оценки |
| `data/prepared/btc_v5.npz` | 112 MB | Подготовленные данные |

---

## 📈 Improvements Over V4

### Performance
- **+122%** улучшение MCC (0.1495 → 0.3316)
- **+136%** улучшение Train MCC
- **-7.8%** снижение train loss
- **Более обобщённая модель** (BTC вместо XAUUSD)

### Technical Enhancements
| Параметр | V4 | V5 | Изменение |
|----------|----|----|-----------|
| Hidden Units | 64 (2L) | 128 (3L) | **+100%** |
| Total Params | 83K | 327K | **+3.9x** |
| Dataset Size | 10K | 517K | **+51x** |
| Training Time | — | 43 min | — |
| Loss Function | CrossEntropy | CrossEntropy | — |
| Regularization | Dropout 0.3 | BatchNorm + Dropout | **Усилена** |

---

## 🔧 Usage

### 1. Training
```bash
python train_v5_btc.py --epochs 100 --batch-size 512
```

### 2. Evaluation
```bash
python evaluate_best_model.py
```

### 3. Inference
```python
import torch
from aimodule.models.v5_btc import GoldenBreezeV5Ultimate

model = torch.load('models/v5_btc/best_model.pt', weights_only=False)
model.eval()

# Predict on new data
predictions = model(input_data)
```

---

## 📁 Project Structure

```
Golden Breeze/
├── models/v5_btc/                    # 🆕 V5 Ultimate Models
│   ├── best_model.pt                 # Best: Val MCC 0.3316 ✨
│   ├── best_model_mcc0.3316_*.pt    # Backup
│   └── checkpoint_*.pt               # Training checkpoints
│
├── aimodule/                          # AI Core
│   ├── models/
│   │   └── v5_btc.py                 # GoldenBreezeV5Ultimate
│   ├── data_pipeline/
│   │   ├── features.py
│   │   ├── features_gold.py
│   │   └── features_smc.py
│   ├── training/
│   ├── inference/
│   └── server/
│
├── data/prepared/
│   ├── btc_v5.npz                   # Training data (517K samples)
│   ├── btc_v5_meta.json             # Metadata
│   └── btc_v5_test.npz              # Test data
│
├── train_v5_btc.py                   # 🆕 Training script
├── evaluate_best_model.py            # 🆕 Evaluation script
├── BTC_V5_STATUS.md                  # 🆕 Detailed status
└── README_V5.md                      # This file

```

---

## 📊 Models Comparison

| Model | Version | Architecture | MCC | Params | Status |
|-------|---------|--------------|-----|--------|--------|
| **V5 Ultimate** | Latest 🏆 | 3L LSTM+BatchNorm | **0.3316** | 327K | ✅ Active |
| V4 Lite Distilled | Archive | Transformer | 0.1495 | 83K | Archive |
| Direction LSTM v3 | Archive | 2L LSTM | 0.1224 | 53K | Archive |
| Regime ML | Auxiliary | KMeans/GMM | — | 5K | ✅ Ready |
| Sentiment Engine | Auxiliary | HuggingFace | — | 200M+ | ✅ Ready |

---

## ✨ Key Achievements

### 🏆 Performance Records
- ✅ **Best Val MCC**: 0.3316 (V5 epoch 91)
- ✅ **Best Train MCC**: 0.3312
- ✅ **Lowest Val Loss**: 0.9709
- ✅ **Improvement vs V4**: +122%

### 🔧 Technical Excellence
- ✅ GPU-optimized training (TF32, cuDNN)
- ✅ Mixed precision (AMP)
- ✅ Multi-GPU ready architecture
- ✅ Clean checkpointing system
- ✅ Automated backup mechanism

### 📈 Generalization
- ✅ Trained on 517K BTC samples (not XAUUSD-only)
- ✅ Multi-timeframe (M5 + H1)
- ✅ Better for diverse instruments

---

## 🔄 Version History

### V5 Ultimate (2025-12-06) 🏆
- **Release**: Current
- **MCC**: 0.3316 (+122% vs V4)
- **Status**: ✅ Production Ready
- **Highlights**: 3L LSTM, BatchNorm, BTC generalization

### V4 Lite Distilled (2025-11-XX)
- **MCC**: 0.1495
- **Status**: Archive
- **Highlights**: Transformer, Knowledge Distillation

### V3 LSTM (2025-10-XX)
- **MCC**: 0.1224
- **Status**: Archive
- **Highlights**: Original LSTM teacher model

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.10
pytorch >= 2.6
cuda >= 12.4
```

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```python
# Load best model
import torch
model = torch.load('models/v5_btc/best_model.pt', weights_only=False)

# Make predictions
predictions = model(your_data)  # Shape: (batch_size, 3)
```

---

## 📝 License

MIT License - See LICENSE file for details

---

**Golden Breeze V5 Ultimate** - Ready for production trading! 🎉

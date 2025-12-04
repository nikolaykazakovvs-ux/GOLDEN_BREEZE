"""
Train V4 Lite using V3 LSTM predictions (Knowledge Distillation)

Упрощённая версия: v4 Lite учится у v3 LSTM.

Author: Golden Breeze Team
Version: 4.2.1
Date: 2025-12-04
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.direction_lstm_model import DirectionLSTMModel
from aimodule.models.v4_transformer.config_lite import V4LiteConfig
from aimodule.models.v4_transformer.model_lite import GoldenBreezeLite
from aimodule.data_pipeline.features_v3 import V3Features
from aimodule.data_pipeline.strategy_signals import StrategySignalsGenerator


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss."""
    
    def __init__(self, alpha: float = 0.3, temperature: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.hard_loss_fn(student_logits, labels)
        
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.soft_loss_fn(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def load_v3_teacher(model_path: str, config_path: str, device: torch.device):
    """Load trained v3 LSTM model."""
    print(f"📚 Loading v3 teacher from {model_path}...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = DirectionLSTMModel(
        input_size=config.get('n_features', 11),
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        num_classes=config.get('n_classes', 2),
        head_type='single_layer',
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Teacher loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   Best MCC: {config.get('best_val_mcc', 'N/A'):.4f}")
    
    return model, config


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""
    
    def __init__(
        self,
        features: np.ndarray,         # v4 features for student (n_samples, 15)
        teacher_features: np.ndarray, # v3 features for teacher (n_samples, 11)
        labels: np.ndarray,
        teacher_model: DirectionLSTMModel,
        device: torch.device,
        seq_len: int = 50,
        strategy_signals: np.ndarray = None,
        smc_static: np.ndarray = None,
    ):
        self.features = features.astype(np.float32)
        self.teacher_features = teacher_features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.teacher = teacher_model
        self.device = device
        self.seq_len = seq_len
        self.strategy_signals = strategy_signals
        self.smc_static = smc_static
        
        # Pre-compute teacher logits using TEACHER features (v3)
        self.teacher_logits = self._compute_teacher_logits()
        
        # Valid indices
        self.valid_indices = []
        for i in range(seq_len, len(self.features)):
            if self.labels[i] >= 0:
                self.valid_indices.append(i)
    
    def _compute_teacher_logits(self):
        """Pre-compute all teacher predictions using v3 features."""
        print("🎓 Pre-computing teacher predictions...")
        self.teacher.eval()
        
        n_samples = len(self.teacher_features) - self.seq_len
        all_logits = np.zeros((n_samples, 2), dtype=np.float32)
        
        batch_size = 256
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                
                batch_seqs = []
                for i in range(start, end):
                    # Use TEACHER features (v3, 11 features)
                    seq = self.teacher_features[i:i + self.seq_len]
                    batch_seqs.append(seq)
                
                x = torch.tensor(np.stack(batch_seqs), dtype=torch.float32).to(self.device)
                logits = self.teacher(x)
                all_logits[start:end] = logits.cpu().numpy()
        
        preds = np.argmax(all_logits, axis=1)
        print(f"   Teacher predictions: DOWN={np.sum(preds==0)}, UP={np.sum(preds==1)}")
        
        return all_logits
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # Student input sequence
        x = torch.tensor(self.features[real_idx - self.seq_len:real_idx], dtype=torch.float32)
        
        # True label
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        
        # Teacher logits
        teacher_idx = real_idx - self.seq_len
        teacher_logit = torch.tensor(self.teacher_logits[teacher_idx], dtype=torch.float32)
        
        result = {
            'x': x,
            'label': label,
            'teacher_logits': teacher_logit,
        }
        
        if self.strategy_signals is not None:
            result['strategy_signals'] = torch.tensor(
                self.strategy_signals[real_idx], dtype=torch.float32
            )
        
        if self.smc_static is not None:
            result['smc_static'] = torch.tensor(
                self.smc_static[real_idx], dtype=torch.float32
            )
        
        return result
    
    def get_labels(self):
        return [self.labels[i] for i in self.valid_indices]


def prepare_v3_features(df: pd.DataFrame) -> np.ndarray:
    """Extract v3-compatible features."""
    feature_gen = V3Features()
    # Use only the first 11 features that v3 expects
    all_features = feature_gen.extract_features(df)
    
    # v3 uses these 11: close, returns, log_returns, sma_fast, sma_slow, sma_ratio, atr, atr_norm, rsi, bb_position, volume_ratio
    v3_cols = [
        'close', 'returns', 'log_returns',
        'sma_fast', 'sma_slow', 'sma_ratio',
        'atr', 'atr_norm', 'rsi', 'bb_position', 'volume_ratio'
    ]
    
    return all_features[v3_cols].values.astype(np.float32)


def train_distillation(
    m5_path: str,
    labels_path: str,
    teacher_model_path: str,
    teacher_config_path: str,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 64,
    alpha: float = 0.3,
    temperature: float = 3.0,
    lr: float = 1e-3,
    patience: int = 10,
):
    """Train v4 Lite using knowledge distillation from v3."""
    print("=" * 70)
    print("V4 Lite - Knowledge Distillation from V3")
    print("=" * 70)
    print(f"\n📋 Distillation Config:")
    print(f"   Alpha (hard label weight): {alpha}")
    print(f"   Temperature: {temperature}")
    print(f"   Learning rate: {lr}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Load teacher
    teacher, teacher_config = load_v3_teacher(
        teacher_model_path, teacher_config_path, device
    )
    seq_len = teacher_config.get('seq_len', 50)
    
    # Load data
    print(f"\n📂 Loading M5 data...")
    df_m5 = pd.read_csv(m5_path)
    df_m5['time'] = pd.to_datetime(df_m5['time'])
    df_m5 = df_m5.sort_values('time').reset_index(drop=True)
    print(f"   {len(df_m5)} bars")
    
    print(f"📂 Loading labels...")
    df_labels = pd.read_csv(labels_path)
    df_labels['time'] = pd.to_datetime(df_labels['time'])
    
    # Extract v3 features (11 features)
    print(f"\n🔧 Extracting v3 features...")
    v3_features = prepare_v3_features(df_m5)
    print(f"   Features shape: {v3_features.shape}")
    
    # Also get v4 Lite features (15 features + strategy signals)
    print(f"🔧 Extracting v4 Lite features...")
    v4_feature_gen = V3Features()
    v4_features = v4_feature_gen.extract_features(df_m5).values.astype(np.float32)
    print(f"   V4 features shape: {v4_features.shape}")
    
    # Strategy signals
    print(f"🎯 Generating strategy signals...")
    strat_gen = StrategySignalsGenerator()
    strategy_signals = strat_gen.generate_all_signals(df_m5).values.astype(np.float32)
    print(f"   Strategy signals: {strategy_signals.shape}")
    
    # Merge labels
    merged = df_m5[['time']].merge(
        df_labels[['time', 'direction_label']],
        on='time', how='left'
    )
    raw_labels = merged['direction_label'].fillna(-1).values.astype(np.int64)
    
    # Convert to 2 classes
    print(f"\n🔄 Converting to 2 classes...")
    labels = np.full_like(raw_labels, -1)
    labels[raw_labels == 0] = 0  # DOWN
    labels[raw_labels == 2] = 1  # UP
    
    valid = labels[labels >= 0]
    print(f"   Valid: {len(valid)} (DOWN: {(valid==0).sum()}, UP: {(valid==1).sum()})")
    
    # Create dataset with v3 features for teacher
    # But we'll also need v4 features for student
    # Solution: store both feature sets
    
    # Simple SMC static features
    print(f"\n📊 Creating SMC static features...")
    smc_static = np.zeros((len(df_m5), 8), dtype=np.float32)
    close = df_m5['close'].values
    high = df_m5['high'].values
    low = df_m5['low'].values
    
    for i in range(len(df_m5)):
        start_idx = max(0, i - 50)
        rolling_high = high[start_idx:i+1].max() if i > 0 else high[i]
        rolling_low = low[start_idx:i+1].min() if i > 0 else low[i]
        
        smc_static[i, 0] = (close[i] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        smc_static[i, 1] = (rolling_high - close[i]) / (rolling_high - rolling_low + 1e-8)
        if smc_static.shape[1] > 2:
            smc_static[i, 2] = (high[i] - low[i]) / (close[i] + 1e-8)
    
    # Create dataset with BOTH v4 (student) and v3 (teacher) features
    print(f"\n📦 Creating distillation dataset...")
    dataset = DistillationDataset(
        features=v4_features,          # v4 features for student (15 features)
        teacher_features=v3_features,  # v3 features for teacher (11 features)
        labels=labels,
        teacher_model=teacher,
        device=device,
        seq_len=seq_len,
        strategy_signals=strategy_signals,
        smc_static=smc_static,
    )
    
    # Split
    n = len(dataset)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n))
    
    print(f"\n📊 Splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)})")
    
    # Create weighted sampler
    train_labels = [dataset.labels[dataset.valid_indices[i]] for i in train_indices]
    label_counts = Counter(train_labels)
    n_train = len(train_labels)
    
    class_weights = {0: n_train / (2 * label_counts.get(0, 1)),
                     1: n_train / (2 * label_counts.get(1, 1))}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"⚖️ Class weights: DOWN={class_weights[0]:.2f}, UP={class_weights[1]:.2f}")
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create student model (v4 Lite)
    print(f"\n🏗️ Creating v4 Lite student...")
    config = V4LiteConfig()
    student = GoldenBreezeLite(config).to(device)
    
    print(f"   Student params: {sum(p.numel() for p in student.parameters()):,}")
    
    # Training
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    print(f"\n   Loss: DistillationLoss (α={alpha}, T={temperature})")
    print(f"   Optimizer: Adam (lr={lr})")
    
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    best_mcc = -1.0
    # patience parameter is passed from args
    patience_counter = 0
    history = {'train_loss': [], 'val_mcc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        student.train()
        train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch['x'].to(device)
            labels_batch = batch['label'].to(device)
            teacher_logits_batch = batch['teacher_logits'].to(device)
            
            smc_static_batch = batch.get('smc_static')
            if smc_static_batch is not None:
                smc_static_batch = smc_static_batch.to(device)
            
            strat_batch = batch.get('strategy_signals')
            if strat_batch is not None:
                strat_batch = strat_batch.to(device)
            
            # Student forward
            student_logits = student(x, smc_static_batch, strat_batch)
            
            # Distillation loss
            loss = criterion(student_logits, teacher_logits_batch, labels_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / max(train_steps, 1)
        
        # Validate
        student.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                labels_batch = batch['label']
                
                smc_static_batch = batch.get('smc_static')
                if smc_static_batch is not None:
                    smc_static_batch = smc_static_batch.to(device)
                
                strat_batch = batch.get('strategy_signals')
                if strat_batch is not None:
                    strat_batch = strat_batch.to(device)
                
                logits = student(x, smc_static_batch, strat_batch)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_train_loss)
        history['val_mcc'].append(val_mcc)
        history['val_acc'].append(val_acc)
        
        pred_counts = Counter(all_preds)
        print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
              f"Loss: {avg_train_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | "
              f"Pred: DOWN={pred_counts.get(0,0)}, UP={pred_counts.get(1,0)}")
        
        # Early stopping
        if val_mcc > best_mcc + 0.001:
            best_mcc = val_mcc
            patience_counter = 0
            
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'val_mcc': val_mcc,
                'config': config.__dict__,
                'distillation': {'alpha': alpha, 'temperature': temperature},
            }, os.path.join(output_dir, "v4_lite_distilled.pt"))
            print(f"   ✅ Saved (MCC: {val_mcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping")
                break
    
    # Test evaluation
    print("\n" + "=" * 70)
    print("Test Evaluation")
    print("=" * 70)
    
    checkpoint = torch.load(os.path.join(output_dir, "v4_lite_distilled.pt"))
    student.load_state_dict(checkpoint['model_state_dict'])
    
    student.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            labels_batch = batch['label']
            
            smc_static_batch = batch.get('smc_static')
            if smc_static_batch is not None:
                smc_static_batch = smc_static_batch.to(device)
            
            strat_batch = batch.get('strategy_signals')
            if strat_batch is not None:
                strat_batch = strat_batch.to(device)
            
            logits = student(x, smc_static_batch, strat_batch)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Macro: {test_f1:.4f}")
    print(f"   MCC:      {test_mcc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['DOWN', 'UP'], zero_division=0))
    
    print(f"\n📉 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    teacher_mcc = teacher_config.get('best_val_mcc', 0)
    print(f"\n🎓 Teacher MCC: {teacher_mcc:.4f}")
    print(f"🎯 Student MCC: {test_mcc:.4f}")
    
    if test_mcc > teacher_mcc:
        print("✅ Student BEATS teacher!")
    elif test_mcc > teacher_mcc * 0.8:
        print("📈 Student approaching teacher performance")
    
    return student, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5", default="data/raw/XAUUSD/M5.csv")
    parser.add_argument("--labels", default="data/labels/direction_labels_XAUUSD_6m.csv")
    parser.add_argument("--teacher-model", default="models/direction_lstm_hybrid_XAUUSD.pt")
    parser.add_argument("--teacher-config", default="models/direction_lstm_hybrid_XAUUSD.json")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    
    train_distillation(
        m5_path=args.m5,
        labels_path=args.labels,
        teacher_model_path=args.teacher_model,
        teacher_config_path=args.teacher_config,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        temperature=args.temperature,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()

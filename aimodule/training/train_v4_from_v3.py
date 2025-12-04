"""
Train V4 Transformer using V3 LSTM predictions (Knowledge Distillation)

Идея: v3 LSTM уже хорошо работает (MCC=0.22).
Вместо обучения v4 с нуля на сырых метках,
обучаем v4 на "мягких метках" от v3.

Это называется Knowledge Distillation:
- Teacher: v3 LSTM (уже обучена)
- Student: v4 Transformer (учится у Teacher)

Author: Golden Breeze Team
Version: 4.2.0
Date: 2025-12-04
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
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
from aimodule.models.v4_transformer.config import V4Config
from aimodule.models.v4_transformer.model import GoldenBreezeFusionV4
from aimodule.training.v4_dataset import FusionDatasetV2, create_dataloaders


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    Combines:
    1. Hard loss: CrossEntropy with true labels
    2. Soft loss: KL-divergence with teacher's soft predictions
    
    Loss = alpha * hard_loss + (1 - alpha) * soft_loss
    """
    
    def __init__(self, alpha: float = 0.3, temperature: float = 3.0):
        """
        Args:
            alpha: Weight for hard loss (0-1). Lower = more trust in teacher.
            temperature: Softmax temperature. Higher = softer distributions.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hard_loss_fn = nn.CrossEntropyLoss()
        self.soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,  # (batch, num_classes)
        teacher_logits: torch.Tensor,  # (batch, num_classes)
        labels: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        """
        # Hard loss: student vs true labels
        hard_loss = self.hard_loss_fn(student_logits, labels)
        
        # Soft loss: student vs teacher (with temperature)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.soft_loss_fn(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return loss


def load_v3_teacher(
    model_path: str,
    config_path: str,
    device: torch.device,
) -> DirectionLSTMModel:
    """
    Load trained v3 LSTM model as teacher.
    """
    print(f"📚 Loading v3 teacher from {model_path}...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model with correct parameter names
    model = DirectionLSTMModel(
        input_size=config.get('n_features', 11),
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        num_classes=config.get('n_classes', 2),
        head_type='single_layer',  # Модель сохранена с single_layer head
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Teacher is always in eval mode
    
    print(f"   ✅ Loaded v3 teacher: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"   Config: hidden={config.get('hidden_size')}, layers={config.get('num_layers')}")
    print(f"   Best MCC: {config.get('best_val_mcc', 'N/A')}")
    
    return model, config


def prepare_v3_features(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Extract features that v3 model expects.
    
    v3 uses 11 features:
    close, returns, log_returns, sma_fast, sma_slow, sma_ratio,
    atr, atr_norm, rsi, bb_position, volume_ratio
    """
    df = df.copy()
    
    # Get volume column
    vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df[vol_col].astype(float)
    
    # 1. Normalized close
    rolling_mean = close.rolling(window=50, min_periods=1).mean()
    rolling_std = close.rolling(window=50, min_periods=1).std()
    df['close_norm'] = (close - rolling_mean) / (rolling_std + 1e-8)
    
    # 2-3. Returns
    df['returns'] = close.pct_change().fillna(0)
    df['log_returns'] = np.log(close / close.shift(1)).fillna(0)
    
    # 4-6. SMA
    sma_fast = close.rolling(window=10, min_periods=1).mean()
    sma_slow = close.rolling(window=50, min_periods=1).mean()
    df['sma_fast'] = (sma_fast - rolling_mean) / (rolling_std + 1e-8)
    df['sma_slow'] = (sma_slow - rolling_mean) / (rolling_std + 1e-8)
    df['sma_ratio'] = sma_fast / (sma_slow + 1e-8) - 1.0
    
    # 7-8. ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()
    atr_rolling_std = atr.rolling(window=50, min_periods=1).std()
    df['atr'] = (atr - atr.rolling(window=50, min_periods=1).mean()) / (atr_rolling_std + 1e-8)
    df['atr_norm'] = atr / (close + 1e-8)
    
    # 9. RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = (rsi - 50) / 50
    
    # 10. Bollinger Band Position
    bb_sma = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    df['bb_position'] = 2 * (close - bb_lower) / (bb_upper - bb_lower + 1e-8) - 1
    
    # 11. Volume Ratio
    volume_ma = volume.rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = (volume / (volume_ma + 1e-8)) - 1.0
    
    # Select features
    feature_cols = [
        'close_norm', 'returns', 'log_returns',
        'sma_fast', 'sma_slow', 'sma_ratio',
        'atr', 'atr_norm', 'rsi', 'bb_position', 'volume_ratio'
    ]
    
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    features = np.clip(features, -10, 10)
    
    return features


def get_teacher_predictions(
    teacher: DirectionLSTMModel,
    features: np.ndarray,
    seq_len: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Get teacher's soft predictions for all samples.
    
    Returns logits (not probabilities) for distillation.
    """
    print("🎓 Generating teacher predictions...")
    teacher.eval()
    
    all_logits = []
    
    n_samples = len(features) - seq_len
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Create batch of sequences
            batch_seqs = []
            for i in range(start_idx, end_idx):
                seq = features[i:i + seq_len]
                batch_seqs.append(seq)
            
            # Stack and convert to tensor
            x = torch.tensor(np.stack(batch_seqs), dtype=torch.float32).to(device)
            
            # Get teacher logits
            logits = teacher(x)
            all_logits.append(logits.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    print(f"   Generated {len(all_logits)} teacher predictions")
    
    # Print teacher's prediction distribution
    preds = np.argmax(all_logits, axis=1)
    pred_counts = Counter(preds)
    print(f"   Teacher prediction dist: {dict(pred_counts)}")
    
    return all_logits


class DistillationDataset(Dataset):
    """
    Dataset for knowledge distillation.
    
    Returns:
    - v4 inputs (M5 sequence, H1 sequence, SMC, strategy signals)
    - Teacher logits from v3
    - True labels
    """
    
    def __init__(
        self,
        v4_dataset: FusionDatasetV2,
        teacher_logits: np.ndarray,
        v3_seq_len: int = 50,
    ):
        """
        Args:
            v4_dataset: FusionDatasetV2 instance
            teacher_logits: Teacher's logits array
            v3_seq_len: v3's sequence length for alignment
        """
        self.v4_dataset = v4_dataset
        self.teacher_logits = teacher_logits
        self.v3_seq_len = v3_seq_len
    
    def __len__(self) -> int:
        return len(self.v4_dataset)
    
    def __getitem__(self, idx: int):
        # Get v4 sample
        v4_sample = self.v4_dataset[idx]
        
        # Get corresponding teacher logits
        # Align indices between v4 and teacher
        row = self.v4_dataset.aligned_meta.iloc[idx]
        m5_idx = int(row['m5_idx'])
        
        # Teacher index (accounting for sequence start)
        teacher_idx = m5_idx - self.v3_seq_len
        
        if 0 <= teacher_idx < len(self.teacher_logits):
            teacher_logit = torch.tensor(
                self.teacher_logits[teacher_idx],
                dtype=torch.float32
            )
        else:
            # Fallback: uniform distribution
            teacher_logit = torch.zeros(2, dtype=torch.float32)
        
        v4_sample['teacher_logits'] = teacher_logit
        
        return v4_sample


def train_with_distillation(
    m5_path: str,
    h1_path: str,
    labels_path: str,
    teacher_model_path: str,
    teacher_config_path: str,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    alpha: float = 0.3,  # Weight for hard labels (lower = more trust in teacher)
    temperature: float = 3.0,
    learning_rate: float = 1e-4,
):
    """
    Train v4 using knowledge distillation from v3.
    """
    print("=" * 70)
    print("Golden Breeze v4 - Knowledge Distillation from v3")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Alpha (hard label weight): {alpha}")
    print(f"   Temperature: {temperature}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Step 1: Load v3 teacher
    teacher, teacher_config = load_v3_teacher(
        teacher_model_path, teacher_config_path, device
    )
    v3_seq_len = teacher_config.get('seq_len', 50)
    n_classes_teacher = teacher_config.get('n_classes', 2)
    
    # Step 2: Load M5 data and generate teacher predictions
    print(f"\n📂 Loading M5 data from {m5_path}...")
    df_m5 = pd.read_csv(m5_path)
    df_m5['time'] = pd.to_datetime(df_m5['time'])
    df_m5 = df_m5.sort_values('time').reset_index(drop=True)
    print(f"   Loaded {len(df_m5)} M5 bars")
    
    # Extract v3 features
    print("\n🔧 Extracting v3 features...")
    v3_features = prepare_v3_features(df_m5, teacher_config)
    print(f"   Features shape: {v3_features.shape}")
    
    # Get teacher predictions
    teacher_logits = get_teacher_predictions(
        teacher, v3_features, v3_seq_len, device
    )
    
    # Step 3: Create v4 dataset
    print(f"\n📂 Loading H1 data from {h1_path}...")
    df_h1 = pd.read_csv(h1_path)
    df_h1['time'] = pd.to_datetime(df_h1['time'])
    print(f"   Loaded {len(df_h1)} H1 bars")
    
    print(f"\n📂 Loading labels from {labels_path}...")
    df_labels = pd.read_csv(labels_path)
    df_labels['time'] = pd.to_datetime(df_labels['time'])
    print(f"   Loaded {len(df_labels)} labels")
    
    # Create v4 config with 2 classes (to match v3 teacher)
    v4_config = V4Config()
    v4_config.num_classes = n_classes_teacher
    print(f"\n📐 V4 Config: num_classes={v4_config.num_classes}")
    
    print("\n🔧 Creating v4 dataset...")
    v4_dataset = FusionDatasetV2.from_dataframes(
        df_m5=df_m5,
        df_h1=df_h1,
        df_labels=df_labels,
        config=v4_config,
        label_col='direction_label',
    )
    print(f"   Dataset size: {len(v4_dataset)}")
    
    # Convert 3-class labels to 2-class (skip HOLD)
    if v4_dataset.labels is not None:
        print("\n🔄 Converting labels to 2 classes (skipping HOLD)...")
        new_labels = np.full_like(v4_dataset.labels, -1)
        new_labels[v4_dataset.labels == 0] = 0  # DOWN → 0
        new_labels[v4_dataset.labels == 2] = 1  # UP → 1
        v4_dataset.labels = new_labels
        
        valid = new_labels[new_labels >= 0]
        print(f"   Valid labels: {len(valid)}")
        print(f"   DOWN: {(valid == 0).sum()}, UP: {(valid == 1).sum()}")
    
    # Create distillation dataset
    distill_dataset = DistillationDataset(
        v4_dataset=v4_dataset,
        teacher_logits=teacher_logits,
        v3_seq_len=v3_seq_len,
    )
    
    # Step 4: Create dataloaders
    print("\n📦 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        v4_dataset,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        use_weighted_sampler=True,
    )
    
    # Step 5: Create v4 student model
    print("\n🏗️ Creating v4 student model...")
    student = GoldenBreezeFusionV4(v4_config).to(device)
    
    total_params = sum(p.numel() for p in student.parameters())
    print(f"   Student parameters: {total_params:,}")
    
    # Step 6: Setup training
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\n   Loss: DistillationLoss (alpha={alpha}, T={temperature})")
    print(f"   Optimizer: AdamW (lr={learning_rate})")
    
    # Step 7: Training loop
    print("\n" + "=" * 70)
    print("Training with Knowledge Distillation")
    print("=" * 70)
    
    best_mcc = -1.0
    patience = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mcc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        student.train()
        train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Get inputs
            x_fast = batch['x_fast'].to(device)
            x_slow = batch['x_slow'].to(device)
            smc_static = batch['smc_static'].to(device)
            smc_dynamic = batch['smc_dynamic'].to(device)
            labels = batch['label'].to(device)
            
            # Skip invalid labels
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            x_fast = x_fast[valid_mask]
            x_slow = x_slow[valid_mask]
            smc_static = smc_static[valid_mask]
            smc_dynamic = smc_dynamic[valid_mask]
            labels = labels[valid_mask]
            
            strategy_signals = batch.get('strategy_signals')
            if strategy_signals is not None:
                strategy_signals = strategy_signals.to(device)[valid_mask]
            
            # Get student predictions
            student_logits = student(
                x_fast, x_slow, smc_static, smc_dynamic, strategy_signals
            )
            
            # Generate teacher logits on-the-fly (simpler approach)
            # Use v3 features from x_fast (last v3_seq_len timesteps)
            with torch.no_grad():
                # Extract v3-compatible features from x_fast
                # x_fast is (batch, seq_len_fast, channels)
                # We need to create v3-style features
                batch_size_curr = x_fast.size(0)
                
                # Create pseudo v3 features from x_fast OHLCV
                # This is simplified - in production, use actual v3 feature extraction
                v3_input = x_fast[:, -v3_seq_len:, :min(5, x_fast.size(2))]
                
                # Pad/adjust to match v3's expected input
                if v3_input.size(2) < teacher_config.get('n_features', 11):
                    # Pad with zeros
                    padding = torch.zeros(
                        batch_size_curr, v3_seq_len,
                        teacher_config.get('n_features', 11) - v3_input.size(2),
                        device=device
                    )
                    v3_input = torch.cat([v3_input, padding], dim=2)
                
                teacher_logits_batch = teacher(v3_input)
            
            # Compute distillation loss
            loss = criterion(student_logits, teacher_logits_batch, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        scheduler.step()
        avg_train_loss = train_loss / max(train_steps, 1)
        
        # Validation
        student.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_fast = batch['x_fast'].to(device)
                x_slow = batch['x_slow'].to(device)
                smc_static = batch['smc_static'].to(device)
                smc_dynamic = batch['smc_dynamic'].to(device)
                labels = batch['label'].to(device)
                
                valid_mask = labels >= 0
                if valid_mask.sum() == 0:
                    continue
                
                x_fast = x_fast[valid_mask]
                x_slow = x_slow[valid_mask]
                smc_static = smc_static[valid_mask]
                smc_dynamic = smc_dynamic[valid_mask]
                labels = labels[valid_mask]
                
                strategy_signals = batch.get('strategy_signals')
                if strategy_signals is not None:
                    strategy_signals = strategy_signals.to(device)[valid_mask]
                
                logits = student(
                    x_fast, x_slow, smc_static, smc_dynamic, strategy_signals
                )
                
                # Standard CE loss for validation
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                val_steps += 1
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Metrics
        if len(all_preds) > 0:
            val_acc = accuracy_score(all_labels, all_preds)
            val_mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            val_acc = val_mcc = val_f1 = 0.0
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mcc'].append(val_mcc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        pred_counts = Counter(all_preds)
        print(f"Epoch {epoch+1:3d}/{epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val MCC: {val_mcc:.4f} | "
              f"Pred: {dict(pred_counts)}")
        
        # Early stopping
        if val_mcc > best_mcc + 0.001:
            best_mcc = val_mcc
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "v4_distilled_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'val_acc': val_acc,
                'config': v4_config.__dict__,
                'distillation': {
                    'alpha': alpha,
                    'temperature': temperature,
                    'teacher_mcc': teacher_config.get('best_val_mcc'),
                }
            }, model_path)
            print(f"   ✅ Best model saved (MCC: {val_mcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, "v4_distilled_best.pt"))
    student.load_state_dict(checkpoint['model_state_dict'])
    
    student.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_fast = batch['x_fast'].to(device)
            x_slow = batch['x_slow'].to(device)
            smc_static = batch['smc_static'].to(device)
            smc_dynamic = batch['smc_dynamic'].to(device)
            labels = batch['label'].to(device)
            
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            x_fast = x_fast[valid_mask]
            x_slow = x_slow[valid_mask]
            smc_static = smc_static[valid_mask]
            smc_dynamic = smc_dynamic[valid_mask]
            labels = labels[valid_mask]
            
            strategy_signals = batch.get('strategy_signals')
            if strategy_signals is not None:
                strategy_signals = strategy_signals.to(device)[valid_mask]
            
            logits = student(
                x_fast, x_slow, smc_static, smc_dynamic, strategy_signals
            )
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Macro: {test_f1:.4f}")
    print(f"   MCC:      {test_mcc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['DOWN', 'UP'],
        zero_division=0
    ))
    
    print(f"\n📉 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Save history
    history_path = os.path.join(output_dir, "v4_distilled_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n📁 History saved to {history_path}")
    print(f"\n🎓 Teacher MCC: {teacher_config.get('best_val_mcc', 'N/A'):.4f}")
    print(f"🎯 Student MCC: {test_mcc:.4f}")
    
    return student, history


def main():
    parser = argparse.ArgumentParser(description="Train v4 using Knowledge Distillation from v3")
    parser.add_argument("--m5", type=str, default="data/raw/XAUUSD/M5.csv")
    parser.add_argument("--h1", type=str, default="data/raw/XAUUSD/H1.csv")
    parser.add_argument("--labels", type=str, default="data/labels/direction_labels_XAUUSD_6m.csv")
    parser.add_argument("--teacher-model", type=str, default="models/direction_lstm_hybrid_XAUUSD.pt")
    parser.add_argument("--teacher-config", type=str, default="models/direction_lstm_hybrid_XAUUSD.json")
    parser.add_argument("--output", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="Weight for hard labels (0-1). Lower = more trust in teacher")
    parser.add_argument("--temperature", type=float, default=3.0,
                       help="Distillation temperature. Higher = softer distributions")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_with_distillation(
        m5_path=args.m5,
        h1_path=args.h1,
        labels_path=args.labels,
        teacher_model_path=args.teacher_model,
        teacher_config_path=args.teacher_config,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        temperature=args.temperature,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()

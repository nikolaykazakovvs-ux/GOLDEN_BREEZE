"""
Train Golden Breeze v4 Lite Model

Training script that uses v3-like parameters:
- CrossEntropyLoss instead of FocalLoss
- Adam instead of AdamW with OneCycleLR
- Constant learning rate 1e-3
- Early stopping on MCC
- Batch size 64

Author: Golden Breeze Team
Version: 4.1.0
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v4_transformer.config_lite import V4LiteConfig
from aimodule.models.v4_transformer.model_lite import GoldenBreezeLite
from aimodule.data_pipeline.features_v3 import V3Features
from aimodule.data_pipeline.strategy_signals import StrategySignalsGenerator


class V4LiteDataset(Dataset):
    """
    PyTorch Dataset for v4 Lite training.
    
    Uses v3-style engineered features + optional strategy signals.
    """
    
    def __init__(
        self,
        features: np.ndarray,  # (n_samples, n_features)
        labels: np.ndarray,  # (n_samples,)
        strategy_signals: np.ndarray = None,  # (n_samples, n_signals)
        smc_static: np.ndarray = None,  # (n_samples, smc_dim)
        seq_len: int = 50,
    ):
        """
        Args:
            features: Engineered features array
            labels: Direction labels (0=DOWN, 1=HOLD, 2=UP)
            strategy_signals: Optional strategy signals
            smc_static: Optional SMC static features
            seq_len: Sequence length for training
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.strategy_signals = strategy_signals.astype(np.float32) if strategy_signals is not None else None
        self.smc_static = smc_static.astype(np.float32) if smc_static is not None else None
        self.seq_len = seq_len
        
        # Valid indices (enough history and valid label)
        self.valid_indices = []
        for i in range(seq_len, len(self.features)):
            if self.labels[i] >= 0:  # Valid label
                self.valid_indices.append(i)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        
        # Get sequence
        x = self.features[real_idx - self.seq_len : real_idx]
        x = torch.tensor(x, dtype=torch.float32)
        
        # Get label
        y = torch.tensor(self.labels[real_idx], dtype=torch.long)
        
        result = {'x': x, 'label': y}
        
        # Optional strategy signals (use last value, not sequence)
        if self.strategy_signals is not None:
            strat = torch.tensor(self.strategy_signals[real_idx], dtype=torch.float32)
            result['strategy_signals'] = strat
        
        # Optional SMC static features
        if self.smc_static is not None:
            smc = torch.tensor(self.smc_static[real_idx], dtype=torch.float32)
            result['smc_static'] = smc
        
        return result
    
    def get_labels_for_sampler(self):
        """Get labels for WeightedRandomSampler."""
        return [self.labels[i] for i in self.valid_indices]


def prepare_data(
    m5_path: str,
    labels_path: str,
    config: V4LiteConfig,
    time_col: str = 'time',
    label_col: str = 'direction_label',
):
    """
    Prepare training data.
    
    Args:
        m5_path: Path to M5 OHLCV CSV
        labels_path: Path to labels CSV
        config: V4LiteConfig
        time_col: Time column name
        label_col: Label column name
        
    Returns:
        Tuple of arrays: (features, labels, strategy_signals, smc_static)
    """
    print("=" * 60)
    print("Preparing Data for v4 Lite")
    print("=" * 60)
    
    # Load M5 data
    print(f"\n📂 Loading M5 data from {m5_path}...")
    df_m5 = pd.read_csv(m5_path)
    df_m5[time_col] = pd.to_datetime(df_m5[time_col])
    df_m5 = df_m5.sort_values(time_col).reset_index(drop=True)
    print(f"   Loaded {len(df_m5)} M5 bars")
    
    # Load labels
    print(f"\n📂 Loading labels from {labels_path}...")
    df_labels = pd.read_csv(labels_path)
    df_labels[time_col] = pd.to_datetime(df_labels[time_col])
    print(f"   Loaded {len(df_labels)} labels")
    
    # Step 1: Extract v3-style features
    print("\n🔧 Step 1: Extracting v3-style features...")
    feature_gen = V3Features()
    features_df = feature_gen.extract_features(df_m5)
    features = features_df.values.astype(np.float32)
    print(f"   Features shape: {features.shape}")
    print(f"   Features: {feature_gen.get_feature_names()}")
    
    # Step 2: Generate strategy signals
    strategy_signals = None
    if config.use_strategy_signals:
        print("\n🎯 Step 2: Generating strategy signals...")
        strat_gen = StrategySignalsGenerator()
        strat_df = strat_gen.generate_all_signals(df_m5)
        strategy_signals = strat_df.values.astype(np.float32)
        print(f"   Strategy signals shape: {strategy_signals.shape}")
    
    # Step 3: Create simple SMC static features
    smc_static = None
    if config.static_smc_dim > 0:
        print("\n📊 Step 3: Creating SMC static features...")
        # Simplified SMC: just use high/low relative positions
        smc_static = np.zeros((len(df_m5), config.static_smc_dim), dtype=np.float32)
        
        close = df_m5['close'].values
        high = df_m5['high'].values
        low = df_m5['low'].values
        
        # Rolling high/low distance
        for i in range(len(df_m5)):
            start_idx = max(0, i - 50)
            rolling_high = high[start_idx:i+1].max() if i > 0 else high[i]
            rolling_low = low[start_idx:i+1].min() if i > 0 else low[i]
            
            # Features: distance to high, distance to low, etc.
            smc_static[i, 0] = (close[i] - rolling_low) / (rolling_high - rolling_low + 1e-8)
            smc_static[i, 1] = (rolling_high - close[i]) / (rolling_high - rolling_low + 1e-8)
            if config.static_smc_dim > 2:
                smc_static[i, 2] = (high[i] - low[i]) / (close[i] + 1e-8)  # Bar range
            if config.static_smc_dim > 3:
                smc_static[i, 3] = (close[i] - (high[i] + low[i]) / 2) / ((high[i] - low[i]) + 1e-8)  # Body position
        
        print(f"   SMC static shape: {smc_static.shape}")
    
    # Step 4: Merge labels
    print("\n🏷️ Step 4: Merging labels...")
    df_m5_merged = df_m5[[time_col]].copy()
    df_m5_merged = df_m5_merged.merge(
        df_labels[[time_col, label_col]],
        on=time_col,
        how='left'
    )
    raw_labels = df_m5_merged[label_col].fillna(-1).values.astype(np.int64)
    
    # If using 2 classes: convert 0→0 (DOWN), 1→-1 (HOLD=skip), 2→1 (UP)
    if config.num_classes == 2:
        print("  📌 Using 2 classes (DOWN/UP only, skipping HOLD)")
        labels = np.full_like(raw_labels, -1)
        labels[raw_labels == 0] = 0  # DOWN → 0
        labels[raw_labels == 2] = 1  # UP → 1
        # HOLD (1) stays as -1 = skip
    else:
        labels = raw_labels
    
    valid_labels = labels[labels >= 0]
    print(f"   Total samples: {len(labels)}")
    print(f"   Valid labels: {len(valid_labels)}")
    print(f"   Label distribution:")
    label_names = ['DOWN', 'UP'] if config.num_classes == 2 else ['DOWN', 'HOLD', 'UP']
    for label_val in range(config.num_classes):
        count = (valid_labels == label_val).sum()
        pct = count / len(valid_labels) * 100
        label_name = label_names[label_val]
        print(f"      {label_name}: {count} ({pct:.1f}%)")
    
    return features, labels, strategy_signals, smc_static


def create_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    strategy_signals: np.ndarray,
    smc_static: np.ndarray,
    config: V4LiteConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_weighted_sampler: bool = True,
):
    """
    Create train/val/test DataLoaders.
    """
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split data (no shuffle for time series)
    def split_array(arr, start, end):
        if arr is None:
            return None
        return arr[start:end]
    
    # Train
    train_dataset = V4LiteDataset(
        features[:train_end],
        labels[:train_end],
        split_array(strategy_signals, 0, train_end),
        split_array(smc_static, 0, train_end),
        seq_len=config.seq_len_fast,
    )
    
    # Val
    val_dataset = V4LiteDataset(
        features[train_end:val_end],
        labels[train_end:val_end],
        split_array(strategy_signals, train_end, val_end),
        split_array(smc_static, train_end, val_end),
        seq_len=config.seq_len_fast,
    )
    
    # Test
    test_dataset = V4LiteDataset(
        features[val_end:],
        labels[val_end:],
        split_array(strategy_signals, val_end, None),
        split_array(smc_static, val_end, None),
        seq_len=config.seq_len_fast,
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    # Create weighted sampler for train
    sampler = None
    if use_weighted_sampler and len(train_dataset) > 0:
        train_labels = train_dataset.get_labels_for_sampler()
        label_counts = Counter(train_labels)
        
        # Class weights
        n_samples = len(train_labels)
        class_weights = {}
        for label in range(config.num_classes):
            count = label_counts.get(label, 1)
            class_weights[label] = n_samples / (config.num_classes * count)
        
        # Sample weights
        sample_weights = [class_weights.get(lbl, 1.0) for lbl in train_labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        print(f"\n⚖️ Class weights for training:")
        label_names = ['DOWN', 'UP'] if config.num_classes == 2 else ['DOWN', 'HOLD', 'UP']
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / n_samples * 100
            weight = class_weights.get(label, 1.0)
            label_name = label_names[label] if label < len(label_names) else f"Class{label}"
            print(f"   {label_name}: {count} ({pct:.1f}%) → weight: {weight:.3f}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: GoldenBreezeLite,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: V4LiteConfig,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        optimizer.zero_grad()
        
        x = batch['x'].to(device)
        labels = batch['label'].to(device)
        
        # Optional inputs
        smc_static = batch.get('smc_static')
        if smc_static is not None:
            smc_static = smc_static.to(device)
        
        strategy_signals = batch.get('strategy_signals')
        if strategy_signals is not None:
            strategy_signals = strategy_signals.to(device)
        
        # Forward
        logits = model(x, smc_static, strategy_signals)
        
        # Loss
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
    
    return avg_loss, acc, f1, mcc


def validate(
    model: GoldenBreezeLite,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: V4LiteConfig,
):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            labels = batch['label'].to(device)
            
            smc_static = batch.get('smc_static')
            if smc_static is not None:
                smc_static = smc_static.to(device)
            
            strategy_signals = batch.get('strategy_signals')
            if strategy_signals is not None:
                strategy_signals = strategy_signals.to(device)
            
            logits = model(x, smc_static, strategy_signals)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_preds)) > 1 else 0.0
    
    return avg_loss, acc, f1, mcc, all_preds, all_labels


def train(
    m5_path: str,
    labels_path: str,
    output_dir: str = "models",
    config: V4LiteConfig = None,
):
    """
    Main training function.
    """
    print("=" * 60)
    print("Golden Breeze v4 Lite Training")
    print("=" * 60)
    
    config = config or V4LiteConfig()
    print(config.summary())
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {device}")
    
    # Prepare data
    features, labels, strategy_signals, smc_static = prepare_data(
        m5_path, labels_path, config
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        features, labels, strategy_signals, smc_static, config
    )
    
    # Create model
    print("\n🏗️ Creating model...")
    model = GoldenBreezeLite(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss function (v3-style: simple CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    print(f"   Loss: CrossEntropyLoss (like v3)")
    
    # Optimizer (v3-style: simple Adam)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    print(f"   Optimizer: Adam (lr={config.learning_rate})")
    
    # Training
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)
    
    best_mcc = -1.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_mcc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_mcc': [],
    }
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_f1, train_mcc = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_mcc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, config
        )
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_mcc'].append(train_mcc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_mcc'].append(val_mcc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}, MCC={train_mcc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}, MCC={val_mcc:.4f}")
        
        # Check class distribution in predictions
        pred_counts = Counter(val_preds)
        if config.num_classes == 2:
            print(f"  Pred dist: DOWN={pred_counts.get(0, 0)}, UP={pred_counts.get(1, 0)}")
        else:
            print(f"  Pred dist: DOWN={pred_counts.get(0, 0)}, HOLD={pred_counts.get(1, 0)}, UP={pred_counts.get(2, 0)}")
        
        # Early stopping on MCC (like v3)
        if val_mcc > best_mcc + config.min_delta:
            best_mcc = val_mcc
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "v4_lite_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config.__dict__,
            }, model_path)
            print(f"  ✅ Best model saved (MCC: {val_mcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total training time: {total_time/60:.1f} minutes")
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, "v4_lite_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_mcc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, config
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 Macro: {test_f1:.4f}")
    print(f"   MCC:      {test_mcc:.4f}")
    
    print(f"\n📋 Classification Report:")
    target_names = ['DOWN', 'UP'] if config.num_classes == 2 else ['DOWN', 'HOLD', 'UP']
    print(classification_report(
        test_labels, test_preds,
        target_names=target_names,
        zero_division=0
    ))
    
    print(f"\n📉 Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    
    # Save history
    history_path = os.path.join(output_dir, "v4_lite_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📁 History saved to {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Golden Breeze v4 Lite")
    parser.add_argument("--m5", type=str, default="data/raw/XAUUSD/M5.csv",
                       help="Path to M5 OHLCV CSV")
    parser.add_argument("--labels", type=str, default="data/labels/direction_labels.csv",
                       help="Path to labels CSV")
    parser.add_argument("--output", type=str, default="models",
                       help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--no-strategy", action="store_true",
                       help="Disable strategy signals")
    
    args = parser.parse_args()
    
    # Create config
    config = V4LiteConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_strategy_signals=not args.no_strategy,
    )
    
    # Train
    train(
        m5_path=args.m5,
        labels_path=args.labels,
        output_dir=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()

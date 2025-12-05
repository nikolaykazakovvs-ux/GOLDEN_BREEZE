"""
Golden Breeze V4 - LSTM Training Script (3-Class)

Training script for LSTMModelV4 with 5-class to 3-class mapping.

Label Mapping:
    0 (Strong Down), 1 (Weak Down) -> 0 (DOWN)
    2 (Neutral)                    -> 1 (NEUTRAL)
    3 (Weak Up), 4 (Strong Up)     -> 2 (UP)

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import matthews_corrcoef, confusion_matrix

# Set multiprocessing start method and sharing strategy for Windows
try:
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')  # Use file_system instead of file_descriptor on Windows
except RuntimeError:
    pass

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v4_lstm import LSTMModelV4, LSTMConfig


# =============================================================================
# DATASET WITH 5->3 CLASS MAPPING
# =============================================================================

class MappedDataset(Dataset):
    """
    Dataset that loads 5-class data and maps to 3 classes on-the-fly.
    
    Mapping:
        0,1 -> 0 (DOWN)
        2   -> 1 (NEUTRAL)
        3,4 -> 2 (UP)
    """
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(self, npz_path: str, preload_to_gpu: bool = False, device: str = 'cuda'):
        print(f"📂 Loading dataset from {npz_path}...")
        
        # Load to RAM as numpy arrays first
        data = np.load(npz_path, allow_pickle=True)
        
        self.preload_to_gpu = preload_to_gpu
        self.device = device
        
        if preload_to_gpu and torch.cuda.is_available():
            print(f"   🚀 PRELOADING ENTIRE DATASET TO GPU (using shared memory)...")
            # Load directly to GPU - uses dedicated + shared VRAM!
            self.x_fast = torch.from_numpy(data['x_fast']).float().to(device)
            self.x_slow = torch.from_numpy(data['x_slow']).float().to(device)
            self.x_strategy = torch.from_numpy(data['x_strategy']).float().to(device)
        else:
            print(f"   Converting to torch tensors (shared memory)...")
            # Convert to torch tensors ONCE (shared across workers)
            # Using .share_memory_() to enable zero-copy sharing between processes
            self.x_fast = torch.from_numpy(data['x_fast']).float().share_memory_()
            self.x_slow = torch.from_numpy(data['x_slow']).float().share_memory_()
            self.x_strategy = torch.from_numpy(data['x_strategy']).float().share_memory_()
        
        # Map 5-class to 3-class
        labels_5class = torch.from_numpy(data['y']).long()
        labels_3class = torch.where(
            labels_5class <= 1, 
            torch.tensor(0, dtype=torch.long),
            torch.where(labels_5class == 2, torch.tensor(1, dtype=torch.long), torch.tensor(2, dtype=torch.long))
        )
        
        if preload_to_gpu and torch.cuda.is_available():
            self.labels = labels_3class.to(device)
            mem_type = "GPU (dedicated + shared)"
        else:
            self.labels = labels_3class.share_memory_()
            mem_type = "shared CPU memory"
        
        self.n_samples = len(self.labels)
        
        print(f"   ✅ Loaded to {mem_type}: {self.n_samples:,} samples")
        print(f"   x_fast: {tuple(self.x_fast.shape)}")
        print(f"   x_slow: {tuple(self.x_slow.shape)}")
        print(f"   x_strategy: {tuple(self.x_strategy.shape)}")
        
        # Print 3-class distribution
        print(f"\n📊 3-Class Distribution:")
        for i, name in enumerate(self.CLASS_NAMES):
            count = (self.labels == i).sum().item()
            pct = count / self.n_samples * 100
            print(f"   {name}: {count:,} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Data already in torch tensors (shared memory), just return slices
        return {
            'x_fast': self.x_fast[idx],
            'x_slow': self.x_slow[idx],
            'x_strat': self.x_strategy[idx],
            'label': self.labels[idx],
        }
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        counts = torch.bincount(self.labels, minlength=3)
        weights = self.n_samples / (3 * counts.float())
        return weights


class DatasetSubset(Dataset):
    """Subset wrapper for train/val/test splits."""
    
    def __init__(self, parent: MappedDataset, indices: list):
        self.parent = parent
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.parent[self.indices[idx]]
    
    def get_labels(self) -> torch.Tensor:
        return self.parent.labels[self.indices]


def create_splits(
    dataset: MappedDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[DatasetSubset, DatasetSubset, DatasetSubset]:
    """Split dataset into train/val/test (time-ordered)."""
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return (
        DatasetSubset(dataset, list(range(0, train_end))),
        DatasetSubset(dataset, list(range(train_end, val_end))),
        DatasetSubset(dataset, list(range(val_end, n))),
    )


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute accuracy, MCC, and per-class accuracy."""
    accuracy = (y_true == y_pred).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    
    per_class = {}
    for c in range(3):
        mask = y_true == c
        if mask.sum() > 0:
            per_class[c] = (y_pred[mask] == c).mean()
        else:
            per_class[c] = 0.0
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'per_class': per_class,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 2,  # Gradient accumulation for 2x effective batch
) -> Tuple[float, float]:
    """Train one epoch with gradient accumulation. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()  # Zero once at start
    
    for i, batch in enumerate(loader):
        x_fast = batch['x_fast'].to(device, non_blocking=True)
        x_slow = batch['x_slow'].to(device, non_blocking=True)
        x_strat = batch['x_strat'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Forward with autocast for mixed precision
        logits = model(x_fast, x_slow, x_strat)
        loss = criterion(logits, labels) / accumulation_steps  # Scale loss
        loss.backward()
        
        # Accumulate gradients
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
    """Evaluate model with async data transfer. Returns (loss, metrics, y_true, y_pred)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        x_fast = batch['x_fast'].to(device, non_blocking=True)
        x_slow = batch['x_slow'].to(device, non_blocking=True)
        x_strat = batch['x_strat'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        labels = batch['label'].to(device)
        
        logits = model(x_fast, x_slow, x_strat)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    
    return total_loss / len(y_true), metrics, y_true, y_pred


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
    save_dir: str,
    patience: int = 50,
) -> Dict:
    """Full training loop."""
    os.makedirs(save_dir, exist_ok=True)
    
    best_mcc = -1.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_mcc': [],
        'lr': [],
    }
    
    print("\n" + "=" * 70)
    print("🚀 Starting Long Training Run")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        val_mcc = val_metrics['mcc']
        
        # LR
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mcc'].append(val_mcc)
        history['lr'].append(current_lr)
        
        # Best model check
        marker = ""
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_long_run.pt'))
            
            marker = " ⭐ BEST"
        else:
            patience_counter += 1
        
        # Print every epoch
        per_class_str = " | ".join([
            f"{MappedDataset.CLASS_NAMES[k]}:{v:.2f}" 
            for k, v in val_metrics['per_class'].items()
        ])
        
        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {train_acc:.3f}/{val_acc:.3f} | "
            f"MCC: {val_mcc:+.4f} | "
            f"LR: {current_lr:.2e}{marker}"
        )
        
        # Per-class every 10 epochs
        if epoch % 10 == 0:
            print(f"         Per-class: {per_class_str}")
        
        # Early stopping (but with high patience for long run)
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch} (patience={patience})")
            break
        
        # Checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'history': history,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
            print(f"         💾 Checkpoint saved")
    
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: Epoch {best_epoch}, Val MCC: {best_mcc:+.4f}")
    print("=" * 70)
    
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LSTM V4 (3-Class)")
    parser.add_argument('--data-path', type=str, default='data/prepared/v4_5class_dataset.npz')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=400)  # Benchmark recommended: 400 (80% of max 512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=50)  # Reduced from 100 for 500 epochs
    parser.add_argument('--save-dir', type=str, default='models/v4_lstm')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers (default: 2, 0=single process)')
    parser.add_argument('--preload-gpu', action='store_true', help='Preload entire dataset to GPU (uses shared VRAM)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "=" * 70)
    print("🔥 Golden Breeze V4 - LSTM Long Training (3-Class)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Patience: {args.patience}")
    print()
    
    # Load dataset (preload to GPU if requested - uses shared VRAM!)
    dataset = MappedDataset(args.data_path, preload_to_gpu=args.preload_gpu, device=str(device))
    
    # Splits
    train_ds, val_ds, test_ds = create_splits(dataset)
    print(f"\n📦 Splits: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")
    
    # Class weights
    class_weights = dataset.get_class_weights().to(device)
    print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")
    
    # Weighted sampler
    train_labels_tensor = train_ds.get_labels()
    train_labels = train_labels_tensor.cpu().numpy() if train_labels_tensor.is_cuda else train_labels_tensor.numpy()
    label_counts = Counter(train_labels)
    sample_weights = [1.0 / label_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoaders - if preload_gpu, no workers needed (data already on GPU)
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': not args.preload_gpu,  # No pin_memory if data on GPU
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 16  # Prefetch 16 batches per worker (use free RAM!)
    
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    # Model
    config = LSTMConfig(
        fast_features=dataset.x_fast.shape[2],
        slow_features=dataset.x_slow.shape[2],
        strategy_dim=dataset.x_strategy.shape[1],
        num_classes=3,
    )
    model = LSTMModelV4(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 Model params: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        save_dir=args.save_dir,
        patience=args.patience,
    )
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("📈 Final Test Evaluation")
    print("=" * 70)
    
    # Load best
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_long_run.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MCC: {test_metrics['mcc']:+.4f}")
    print(f"\nPer-Class Accuracy:")
    for k, v in test_metrics['per_class'].items():
        print(f"   {MappedDataset.CLASS_NAMES[k]}: {v:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'epochs_trained': len(history['train_loss']),
        'best_epoch': checkpoint['epoch'],
        'best_val_mcc': float(checkpoint['val_mcc']),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_mcc': float(test_metrics['mcc']),
        'per_class_accuracy': {MappedDataset.CLASS_NAMES[k]: float(v) for k, v in test_metrics['per_class'].items()},
        'confusion_matrix': cm.tolist(),
        'hyperparameters': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'patience': args.patience,
        },
    }
    
    with open(os.path.join(args.save_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to {args.save_dir}/training_report.json")
    print("\n✅ Training Complete!")


if __name__ == "__main__":
    main()

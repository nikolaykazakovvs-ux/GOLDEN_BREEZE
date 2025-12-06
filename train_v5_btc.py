"""
Golden Breeze v5 BTC - Training Script

Trains a dedicated model for BTC/USDT crypto trading.
Based on v5 Ultimate architecture, optimized for crypto volatility.

Usage:
    python train_v5_btc.py --data-path data/prepared/btc_v5.npz

Author: Golden Breeze Team
Version: 5.1.0 BTC
Date: 2025-12-06
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aimodule.models.v5_ultimate import GoldenBreezeV5Ultimate, V5UltimateConfig


# =============================================================================
# DATASET
# =============================================================================

class BTCDataset(Dataset):
    """
    Dataset for BTC v5 training.
    Same as UltimateDataset, loads ALL data into RAM.
    """
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(self, npz_path: str, indices: np.ndarray = None):
        print(f"📂 Loading BTC dataset into RAM: {npz_path}")
        
        # Load with mmap for faster memory-mapped access
        data_dict = np.load(npz_path, mmap_mode='r', allow_pickle=False)
        
        # Don't force copy - use mmap as-is
        x_fast_all = data_dict['x_fast']
        x_slow_all = data_dict['x_slow']
        x_strategy_all = data_dict['x_strategy']
        labels_5class_all = data_dict['y']
        
        if indices is not None:
            x_fast = x_fast_all[indices].copy()
            x_slow = x_slow_all[indices].copy()
            x_strategy = x_strategy_all[indices].copy()
            labels_5class = labels_5class_all[indices].copy()
        else:
            x_fast = x_fast_all[:].copy()
            x_slow = x_slow_all[:].copy()
            x_strategy = x_strategy_all[:].copy()
            labels_5class = labels_5class_all[:].copy()
        
        # Map 5-class to 3-class
        # 0,1 -> 0 (DOWN), 2 -> 1 (NEUTRAL), 3,4 -> 2 (UP)
        labels_3class = np.where(
            labels_5class <= 1, 0,
            np.where(labels_5class == 2, 1, 2)
        )
        
        # Convert to tensors
        self.x_fast = torch.from_numpy(x_fast).float()
        self.x_slow = torch.from_numpy(x_slow).float()
        self.x_strategy = torch.from_numpy(x_strategy).float()
        self.labels = torch.from_numpy(labels_3class).long()
        
        self.n_samples = len(self.labels)
        
        print(f"   ✅ Loaded {self.n_samples:,} samples into RAM")
        print(f"   x_fast: {tuple(self.x_fast.shape)}")
        print(f"   x_slow: {tuple(self.x_slow.shape)}")
        print(f"   x_strategy: {tuple(self.x_strategy.shape)}")
        
        # Class distribution
        print(f"\n📊 Class Distribution:")
        for i, name in enumerate(self.CLASS_NAMES):
            count = (self.labels == i).sum().item()
            pct = count / self.n_samples * 100
            print(f"   {name}: {count:,} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int):
        return (
            self.x_fast[idx],
            self.x_slow[idx],
            self.x_strategy[idx],
            self.labels[idx]
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse class weights for balanced training."""
        counts = torch.bincount(self.labels)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(weights)
        return weights


# =============================================================================
# TRAINING UTILS
# =============================================================================

def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, clip_grad=1.0):
    """Train one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for x_fast, x_slow, x_strat, labels in pbar:
        x_fast = x_fast.to(device, non_blocking=True)
        x_slow = x_slow.to(device, non_blocking=True)
        x_strat = x_strat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward
        with autocast(device_type='cuda'):
            logits = model(x_fast, x_slow, x_strat)
            loss = criterion(logits, labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Track
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return {'loss': avg_loss, 'accuracy': accuracy, 'mcc': mcc}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for x_fast, x_slow, x_strat, labels in tqdm(loader, desc="Evaluating", leave=False):
        x_fast = x_fast.to(device, non_blocking=True)
        x_slow = x_slow.to(device, non_blocking=True)
        x_strat = x_strat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type='cuda'):
            logits = model(x_fast, x_slow, x_strat)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class = {}
    for i in range(3):
        if cm[i].sum() > 0:
            per_class[i] = cm[i, i] / cm[i].sum()
        else:
            per_class[i] = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mcc': mcc,
        'per_class': per_class,
        'confusion_matrix': cm,
    }


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train(args):
    """Main training loop for BTC model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    # Backup existing best_model.pt to avoid losing progress on restart
    best_model_path = Path(args.save_dir) / 'best_model.pt'
    if best_model_path.exists():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(args.save_dir) / f"best_model_backup_{ts}.pt"
        shutil.copy2(best_model_path, backup_path)
        print(f"📦 Backup создан: {backup_path}")

    if device == 'cuda':
        # Разрешаем TF32 для ускорения матриц на Ampere и включаем autotune
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    print("=" * 70)
    print("🪙 Golden Breeze v5 BTC Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load full dataset
    print("\n📂 Loading dataset...")
    data = np.load(args.data_path)
    n_samples = len(data['y'])
    
    # Create splits (70/15/15)
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"   Train: {len(train_idx):,}")
    print(f"   Val: {len(val_idx):,}")
    print(f"   Test: {len(test_idx):,}")
    
    # Create datasets
    train_ds = BTCDataset(args.data_path, train_idx)
    val_ds = BTCDataset(args.data_path, val_idx)
    test_ds = BTCDataset(args.data_path, test_idx)
    
    # Get class weights
    class_weights = train_ds.get_class_weights().to(device)
    print(f"\n⚖️ Class weights: {class_weights.tolist()}")
    
    # Create loaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0,
    }

    # Prefetch ускоряет подачу данных при наличии воркеров
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    # Create model
    config = V5UltimateConfig()
    model = GoldenBreezeV5Ultimate(config).to(device)
    
    total_params = model.count_parameters()
    print(f"\n🔧 Model: GoldenBreezeV5Ultimate (BTC)")
    print(f"   Parameters: {total_params:,}")
    
    # Resume if checkpoint provided
    start_epoch = 1
    best_mcc = -1.0
    best_epoch = 0
    
    # Auto-resume from best_model.pt if --resume не задан, но файл существует
    auto_resume_path = best_model_path if best_model_path.exists() else None
    resume_path = args.resume or auto_resume_path
    if resume_path:
        print(f"\n📂 Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mcc = checkpoint.get('val_mcc', -1.0)
        best_epoch = checkpoint.get('epoch', 0)
        print(f"   Epoch {start_epoch}, best MCC: {best_mcc:+.4f}")
    
    # Optimizer, scheduler, loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = GradScaler()
    
    print(f"\n📅 Scheduler: Warmup ({args.warmup_epochs} epochs) + Cosine")
    print(f"   Total steps: {total_steps:,}")
    
    # Training loop
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("🚀 Starting BTC Training")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, args.clip_grad
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} MCC: {train_metrics['mcc']:+.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} MCC: {val_metrics['mcc']:+.4f} | "
              f"LR: {lr:.2e} | {epoch_time:.1f}s")
        
        # Check improvement
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': best_mcc,
                'val_acc': val_metrics['accuracy'],
                'config': config.__dict__,
                'symbol': 'BTC/USDT',
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"   ✨ New best MCC: {best_mcc:+.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_metrics['mcc'],
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Final test
    print("\n" + "=" * 70)
    print("📈 Final BTC Test Evaluation")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBTC Test Results:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   MCC: {test_metrics['mcc']:+.4f}")
    print(f"\nPer-Class Accuracy:")
    for i, name in enumerate(['DOWN', 'NEUTRAL', 'UP']):
        print(f"   {name}: {test_metrics['per_class'][i]:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Save report
    report = {
        'model': 'GoldenBreezeV5Ultimate_BTC',
        'symbol': 'BTC/USDT',
        'exchange': 'BINANCE',
        'date': datetime.now().isoformat(),
        'total_params': total_params,
        'best_epoch': best_epoch,
        'best_val_mcc': float(best_mcc),
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'mcc': float(test_metrics['mcc']),
            'per_class': {str(k): float(v) for k, v in test_metrics['per_class'].items()},
        },
        'args': vars(args),
    }
    
    with open(os.path.join(args.save_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🪙 BTC Training complete! Results saved to {args.save_dir}/")
    
    return best_mcc


def main():
    parser = argparse.ArgumentParser(description="Train v5 BTC Model")
    parser.add_argument('--data-path', type=str, default='data/prepared/btc_v5.npz')
    parser.add_argument('--save-dir', type=str, default='models/v5_btc')
    parser.add_argument('--batch-size', type=int, default=1024)  # Увеличенный батч для лучшей загрузки GPU
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=4, help='Prefetch batches per worker (>0 when num_workers>0)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from (default: auto if best_model.pt exists)')
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

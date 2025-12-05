"""
Golden Breeze v5 - Training Script for Lite Patch Transformer

Optimized training pipeline with:
- Gradient clipping (transformers can explode)
- Warmup scheduler (gentle start)
- Weight decay (regularization)
- Mixed precision training (faster on GPU)
- Best model saving based on MCC

Author: Golden Breeze Team
Version: 5.0.0
Date: 2025-12-05
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v5_lite_transformer import LitePatchTransformer, V5Config


# =============================================================================
# DATASET
# =============================================================================

class V5Dataset(Dataset):
    """
    Dataset for v5 Lite Transformer.
    
    Loads precomputed features and maps 5-class labels to 3-class.
    Uses memory mapping for efficient large dataset handling.
    """
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(
        self,
        npz_path: str,
        indices: np.ndarray = None,
        preload_gpu: bool = False,
        device: str = 'cuda',
    ):
        """
        Args:
            npz_path: Path to .npz dataset
            indices: Subset indices (for train/val/test split)
            preload_gpu: Whether to preload to GPU
            device: Device for preloading
        """
        print(f"📂 Loading dataset from {npz_path}...")
        
        # Load with memory mapping for efficiency
        data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        
        self.indices = indices
        self.preload_gpu = preload_gpu
        self.device = device
        
        # Get data arrays
        if indices is not None:
            x_fast = data['x_fast'][indices]
            x_strategy = data['x_strategy'][indices]
            labels_5class = data['y'][indices]
        else:
            x_fast = data['x_fast'][:]
            x_strategy = data['x_strategy'][:]
            labels_5class = data['y'][:]
        
        # Map 5-class to 3-class
        # 0,1 -> 0 (DOWN), 2 -> 1 (NEUTRAL), 3,4 -> 2 (UP)
        labels_3class = np.where(
            labels_5class <= 1, 0,
            np.where(labels_5class == 2, 1, 2)
        )
        
        # Convert to tensors
        if preload_gpu and torch.cuda.is_available():
            print(f"   🚀 Preloading to GPU...")
            self.x_fast = torch.from_numpy(x_fast).float().to(device)
            self.x_strategy = torch.from_numpy(x_strategy).float().to(device)
            self.labels = torch.from_numpy(labels_3class).long().to(device)
        else:
            self.x_fast = torch.from_numpy(x_fast.copy()).float()
            self.x_strategy = torch.from_numpy(x_strategy.copy()).float()
            self.labels = torch.from_numpy(labels_3class.copy()).long()
        
        self.n_samples = len(self.labels)
        
        print(f"   ✅ Loaded {self.n_samples:,} samples")
        print(f"   x_fast: {tuple(self.x_fast.shape)}")
        print(f"   x_strategy: {tuple(self.x_strategy.shape)}")
        
        # Print class distribution
        print(f"\n📊 Class Distribution:")
        for i, name in enumerate(self.CLASS_NAMES):
            count = (self.labels == i).sum().item()
            pct = count / self.n_samples * 100
            print(f"   {name}: {count:,} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int):
        return self.x_fast[idx], self.x_strategy[idx], self.labels[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse class weights for balanced training."""
        counts = torch.bincount(self.labels.cpu() if self.preload_gpu else self.labels)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for balanced batches."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels.cpu() if self.preload_gpu else self.labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def get_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    scaler: GradScaler,
    clip_grad: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (x_fast, x_strat, labels) in enumerate(loader):
        # Move to device if not preloaded
        if x_fast.device.type == 'cpu':
            x_fast = x_fast.to(device)
            x_strat = x_strat.to(device)
            labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            logits = model(x_fast, x_strat)
            loss = criterion(logits, labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mcc': mcc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for x_fast, x_strat, labels in loader:
        if x_fast.device.type == 'cpu':
            x_fast = x_fast.to(device)
            x_strat = x_strat.to(device)
            labels = labels.to(device)
        
        with autocast():
            logits = model(x_fast, x_strat)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
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
        'all_preds': np.array(all_preds),
        'all_labels': np.array(all_labels),
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epochs: int,
    save_dir: str,
    patience: int = 30,
    clip_grad: float = 1.0,
):
    """Full training loop with early stopping."""
    
    scaler = GradScaler()
    
    best_mcc = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, clip_grad
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} MCC: {train_metrics['mcc']:+.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} MCC: {val_metrics['mcc']:+.4f} | "
              f"LR: {lr:.2e} | {epoch_time:.1f}s")
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append({k: v for k, v in val_metrics.items() if k not in ['confusion_matrix', 'all_preds', 'all_labels']})
        
        # Check for improvement
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': best_mcc,
                'val_acc': val_metrics['accuracy'],
                'config': model.config.__dict__,
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"   ✨ New best MCC: {best_mcc:+.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (patience={patience})")
                break
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_metrics['mcc'],
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    print("\n" + "=" * 70)
    print(f"✅ Training Complete!")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Best Val MCC: {best_mcc:+.4f}")
    print("=" * 70)
    
    return history, best_mcc, best_epoch


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train v5 Lite Transformer")
    parser.add_argument('--data-path', type=str, default='data/prepared/v4_6year_dataset.npz')
    parser.add_argument('--save-dir', type=str, default='models/v5_lite')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--preload-gpu', action='store_true', help='Preload dataset to GPU')
    parser.add_argument('--num-workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 70)
    print("🚀 Golden Breeze v5 - Lite Patch Transformer Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {args.data_path}")
    print(f"Save dir: {args.save_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Gradient clipping: {args.clip_grad}")
    
    # Load full dataset to get size
    print("\n📂 Loading dataset...")
    data = np.load(args.data_path, mmap_mode='r')
    n_samples = len(data['y'])
    
    # Create train/val/test splits (70/15/15)
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
    train_ds = V5Dataset(args.data_path, train_idx, preload_gpu=args.preload_gpu, device=device)
    val_ds = V5Dataset(args.data_path, val_idx, preload_gpu=args.preload_gpu, device=device)
    test_ds = V5Dataset(args.data_path, test_idx, preload_gpu=args.preload_gpu, device=device)
    
    # Get class weights for loss
    class_weights = train_ds.get_class_weights().to(device)
    print(f"\n⚖️ Class weights: {class_weights.tolist()}")
    
    # Create data loaders
    sampler = train_ds.get_sampler()
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': not args.preload_gpu,
    }
    
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    
    # Create model
    config = V5Config()
    model = LitePatchTransformer(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 Model: LitePatchTransformer")
    print(f"   Parameters: {total_params:,}")
    print(f"   d_model: {config.d_model}")
    print(f"   num_layers: {config.num_layers}")
    print(f"   patch_size: {config.patch_size}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine scheduler
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    print(f"\n📅 Scheduler: Warmup ({args.warmup_epochs} epochs) + Cosine")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    
    # Train
    history, best_mcc, best_epoch = train(
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
        clip_grad=args.clip_grad,
    )
    
    # Final test evaluation
    print("\n" + "=" * 70)
    print("📈 Final Test Evaluation")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   MCC: {test_metrics['mcc']:+.4f}")
    print(f"\nPer-Class Accuracy:")
    for i, name in enumerate(['DOWN', 'NEUTRAL', 'UP']):
        print(f"   {name}: {test_metrics['per_class'][i]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Save final report
    report = {
        'model': 'LitePatchTransformer v5',
        'date': datetime.now().isoformat(),
        'config': config.__dict__,
        'total_params': total_params,
        'best_epoch': best_epoch,
        'best_val_mcc': float(best_mcc),
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'mcc': float(test_metrics['mcc']),
            'per_class': {k: float(v) for k, v in test_metrics['per_class'].items()},
        },
        'args': vars(args),
    }
    
    with open(os.path.join(args.save_dir, 'training_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()

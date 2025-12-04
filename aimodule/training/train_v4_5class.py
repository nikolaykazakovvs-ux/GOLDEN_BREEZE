"""
Golden Breeze V4 - 5-Class Fusion Transformer Training

Training script for the 5-class direction model using precomputed dataset.

Classes:
    0: STRONG_DOWN (< -0.4%)
    1: WEAK_DOWN (-0.4% to -0.1%)
    2: NEUTRAL (-0.1% to +0.1%)
    3: WEAK_UP (+0.1% to +0.4%)
    4: STRONG_UP (> +0.4%)

Author: Golden Breeze Team
Version: 4.2.0 (Focal Loss + Big Data)
Date: 2025-12-04
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v4_transformer.config import V4Config


# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights (Tensor of shape [num_classes])
        gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            logits: (B, C) raw predictions
            targets: (B,) class indices
            
        Returns:
            loss: scalar if reduction='mean'/'sum', else (B,)
        """
        num_classes = logits.size(1)
        
        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            # Smooth cross entropy
            log_probs = F.log_softmax(logits, dim=1)
            
            # Create smoothed targets
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # CE with soft targets
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            
            # Get probabilities for focal modulation
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard CE
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            
            # Get probability of true class
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, targets)
            focal_weight = focal_weight * at
        
        # Final loss
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# DATASET
# =============================================================================

class PrecomputedDataset(Dataset):
    """
    Fast dataset loader for precomputed V4 5-class data.
    
    Loads arrays directly from NPZ:
    - x_fast: M5 sequences (N, 50, 15)
    - x_slow: H1 sequences (N, 20, 8)
    - strategies: Strategy signals (N, 64)
    - labels: 5-class labels (N,)
    """
    
    CLASS_NAMES = ['STRONG_DOWN', 'WEAK_DOWN', 'NEUTRAL', 'WEAK_UP', 'STRONG_UP']
    
    def __init__(self, npz_path: str):
        """
        Load precomputed dataset from NPZ file.
        
        Args:
            npz_path: Path to v4_5class_dataset.npz
        """
        print(f"📂 Loading dataset from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        
        self.x_fast = torch.tensor(data['x_fast'], dtype=torch.float32)
        self.x_slow = torch.tensor(data['x_slow'], dtype=torch.float32)
        self.strategies = torch.tensor(data['x_strategy'], dtype=torch.float32)
        self.labels = torch.tensor(data['y'], dtype=torch.long)
        
        # Get shapes
        self.n_samples = len(self.labels)
        self.seq_len_fast = self.x_fast.shape[1]
        self.seq_len_slow = self.x_slow.shape[1]
        self.fast_features = self.x_fast.shape[2]
        self.slow_features = self.x_slow.shape[2]
        self.strategy_dim = self.strategies.shape[1]
        
        print(f"   Samples: {self.n_samples:,}")
        print(f"   x_fast: {self.x_fast.shape}")
        print(f"   x_slow: {self.x_slow.shape}")
        print(f"   strategies: {self.strategies.shape}")
        print(f"   labels: {self.labels.shape}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x_fast': self.x_fast[idx],
            'x_slow': self.x_slow[idx],
            'strategy_signals': self.strategies[idx],
            'label': self.labels[idx],
        }
    
    def get_class_distribution(self) -> Dict[str, Tuple[int, float]]:
        """Get class distribution."""
        labels_np = self.labels.numpy()
        counts = Counter(labels_np)
        total = len(labels_np)
        
        dist = {}
        for i, name in enumerate(self.CLASS_NAMES):
            count = counts.get(i, 0)
            pct = count / total * 100
            dist[name] = (count, pct)
        
        return dist
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights inversely proportional to frequency.
        
        Formula: weight[c] = N / (num_classes * count[c])
        """
        labels_np = self.labels.numpy()
        counts = np.bincount(labels_np, minlength=5)
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        
        # Inverse frequency weighting
        weights = self.n_samples / (5 * counts)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple['PrecomputedDatasetSubset', 'PrecomputedDatasetSubset', 'PrecomputedDatasetSubset']:
        """
        Split dataset into train/val/test (time-ordered, no shuffle).
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        n = self.n_samples
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, n))
        
        return (
            PrecomputedDatasetSubset(self, train_indices),
            PrecomputedDatasetSubset(self, val_indices),
            PrecomputedDatasetSubset(self, test_indices),
        )


class PrecomputedDatasetSubset(Dataset):
    """Subset of PrecomputedDataset using index mapping."""
    
    def __init__(self, parent: PrecomputedDataset, indices: list):
        self.parent = parent
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.parent[self.indices[idx]]
    
    def get_labels(self) -> torch.Tensor:
        """Get labels for this subset."""
        return self.parent.labels[self.indices]


# =============================================================================
# LIGHTWEIGHT MODEL (for 5-class)
# =============================================================================

class FusionTransformerLite5Class(nn.Module):
    """
    Lightweight Fusion Transformer for 5-class prediction.
    
    Architecture:
    - Fast stream (M5): Linear projection + 1 Transformer layer
    - Slow stream (H1): Linear projection + 1 Transformer layer
    - Strategy MLP: 64 → 64 → 32
    - Fusion: Concatenate + MLP → 5 classes
    """
    
    def __init__(
        self,
        fast_features: int = 15,
        slow_features: int = 8,
        strategy_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Fast stream projection
        self.fast_proj = nn.Linear(fast_features, d_model)
        self.fast_norm = nn.LayerNorm(d_model)
        
        # Slow stream projection
        self.slow_proj = nn.Linear(slow_features, d_model)
        self.slow_norm = nn.LayerNorm(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.fast_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.slow_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=1,
        )
        
        # Strategy MLP
        self.strategy_mlp = nn.Sequential(
            nn.Linear(strategy_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        
        # Fusion dimension: d_model (fast) + d_model (slow) + 32 (strategy)
        fusion_dim = d_model * 2 + 32
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_fast: torch.Tensor,
        x_slow: torch.Tensor,
        strategy_signals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_fast: (B, seq_fast, fast_features)
            x_slow: (B, seq_slow, slow_features)
            strategy_signals: (B, strategy_dim)
            
        Returns:
            logits: (B, num_classes)
        """
        # Fast stream
        fast = self.fast_proj(x_fast)  # (B, seq, d_model)
        fast = self.fast_norm(fast)
        fast = self.fast_encoder(fast)  # (B, seq, d_model)
        fast_pool = fast.mean(dim=1)  # (B, d_model)
        
        # Slow stream
        slow = self.slow_proj(x_slow)  # (B, seq, d_model)
        slow = self.slow_norm(slow)
        slow = self.slow_encoder(slow)  # (B, seq, d_model)
        slow_pool = slow.mean(dim=1)  # (B, d_model)
        
        # Strategy
        strat = self.strategy_mlp(strategy_signals)  # (B, 32)
        
        # Fusion
        fused = torch.cat([fast_pool, slow_pool, strat], dim=1)  # (B, fusion_dim)
        
        # Classification
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits


# =============================================================================
# METRICS
# =============================================================================

def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """
    Compute Matthews Correlation Coefficient for multiclass.
    
    MCC = (c * s - Σ(p_k * t_k)) / sqrt((s² - Σp_k²) * (s² - Σt_k²))
    
    where:
    - c = total correctly classified
    - s = total samples
    - p_k = samples predicted as class k
    - t_k = samples truly belonging to class k
    """
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return (y_true == y_pred).mean()


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> Dict[int, float]:
    """Compute per-class accuracy."""
    result = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            result[c] = (y_pred[mask] == c).mean()
        else:
            result[c] = 0.0
    return result


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        x_fast = batch['x_fast'].to(device)
        x_slow = batch['x_slow'].to(device)
        strategy = batch['strategy_signals'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(x_fast, x_slow, strategy)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model.
    
    Returns:
        (loss, accuracy, mcc, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        x_fast = batch['x_fast'].to(device)
        x_slow = batch['x_slow'].to(device)
        strategy = batch['strategy_signals'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(x_fast, x_slow, strategy)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    accuracy = compute_accuracy(y_true, y_pred)
    mcc = compute_mcc(y_true, y_pred)
    
    return total_loss / len(y_true), accuracy, mcc, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int = 50,
    early_stop_patience: int = 10,
    save_dir: str = 'models/v4_5class',
) -> Dict:
    """
    Full training loop.
    
    Returns:
        Training history dict
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_mcc = -1.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mcc': [],
        'lr': [],
    }
    
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_mcc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Get LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mcc'].append(val_mcc)
        history['lr'].append(current_lr)
        
        # Check for best model
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mcc': val_mcc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_model.pt'))
            
            marker = " ⭐ BEST"
        else:
            patience_counter += 1
            marker = ""
        
        # Print progress
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} MCC: {val_mcc:+.4f}{marker}"
        )
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️ Early stopping at epoch {epoch} (patience={early_stop_patience})")
            break
    
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: Epoch {best_epoch}, Val MCC: {best_val_mcc:+.4f}")
    print("=" * 70)
    
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train V4 5-Class Fusion Transformer")
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/prepared/v4_5class_dataset.npz',
        help='Path to precomputed dataset',
    )
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stop patience')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/v4_5class',
        help='Directory to save model',
    )
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "=" * 70)
    print("🔥 Golden Breeze V4 - 5-Class Fusion Transformer Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stop patience: {args.patience}")
    print()
    
    # Load dataset
    dataset = PrecomputedDataset(args.data_path)
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    dist = dataset.get_class_distribution()
    for name, (count, pct) in dist.items():
        print(f"   {name}: {count:,} ({pct:.1f}%)")
    
    # Class weights
    class_weights = dataset.get_class_weights().to(device)
    print(f"\n⚖️ Class Weights:")
    for i, name in enumerate(PrecomputedDataset.CLASS_NAMES):
        print(f"   {name}: {class_weights[i]:.3f}")
    
    # Split dataset
    train_ds, val_ds, test_ds = dataset.get_splits(train_ratio=0.7, val_ratio=0.15)
    print(f"\n📦 Dataset Splits:")
    print(f"   Train: {len(train_ds):,}")
    print(f"   Val: {len(val_ds):,}")
    print(f"   Test: {len(test_ds):,}")
    
    # Create weighted sampler for training
    train_labels = train_ds.get_labels().numpy()
    label_counts = Counter(train_labels)
    sample_weights = []
    for label in train_labels:
        weight = 1.0 / label_counts[label]
        sample_weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create model
    model = FusionTransformerLite5Class(
        fast_features=dataset.fast_features,
        slow_features=dataset.slow_features,
        strategy_dim=dataset.strategy_dim,
        d_model=64,
        nhead=4,
        num_classes=5,
        dropout=0.2,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Focal Loss with class weights (gamma=3.0 for aggressive focus on hard examples)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=3.0,  # Higher gamma = more focus on misclassified examples
        reduction='mean',
        label_smoothing=0.05,  # Reduced smoothing for Focal Loss
    )
    print(f"\n🎯 Using Focal Loss (γ=3.0)")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    
    # Scheduler: CosineAnnealingWarmRestarts (restart every T_0 epochs)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart LR every 10 epochs
        T_mult=2,  # After restart, next cycle is 2x longer
        eta_min=1e-6,
    )
    print(f"📉 Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        early_stop_patience=args.patience,
        save_dir=args.save_dir,
    )
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("📈 Final Test Evaluation")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_mcc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   MCC: {test_mcc:+.4f}")
    
    # Per-class accuracy
    per_class = compute_per_class_accuracy(y_true, y_pred)
    print(f"\nPer-Class Accuracy:")
    for i, name in enumerate(PrecomputedDataset.CLASS_NAMES):
        print(f"   {name}: {per_class[i]:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_path': args.data_path,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': checkpoint['epoch'],
        'best_val_mcc': float(checkpoint['val_mcc']),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_mcc': float(test_mcc),
        'per_class_accuracy': {
            PrecomputedDataset.CLASS_NAMES[k]: float(v) 
            for k, v in per_class.items()
        },
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            k: {'count': v[0], 'pct': v[1]} 
            for k, v in dataset.get_class_distribution().items()
        },
        'model_params': total_params,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'epochs': args.epochs,
            'patience': args.patience,
        },
    }
    
    report_path = os.path.join(args.save_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved to {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    
    return history, report


if __name__ == "__main__":
    main()

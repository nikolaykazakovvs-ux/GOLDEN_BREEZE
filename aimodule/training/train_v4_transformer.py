"""
Golden Breeze v4 - Training Script for Fusion Transformer

Training Strategy (GPT & Gemini Approved):
- Loss Function:
  * Score Head: HuberLoss (robust regression)
  * Class Head: FocalLoss (imbalanced classification)
  * Total Loss = 1.0 * Huber + 0.5 * Focal
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: OneCycleLR (max_lr=1e-3, pct_start=0.3)
- Gradient Clipping: norm=1.0

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.models.v4_transformer.config import V4Config
from aimodule.models.v4_transformer.model import GoldenBreezeFusionV4, create_dummy_inputs
from aimodule.data_pipeline.smc_processor import SMCProcessor
from aimodule.data_pipeline.alignment import TimeAligner
from aimodule.training.v4_dataset import FusionDatasetV2, FusionDataset, create_dataloaders
from aimodule.training.focal_loss import FocalLoss, DualLoss, ClassificationOnlyLoss


class TrainerV4:
    """
    Trainer class for GoldenBreezeFusionV4 model.
    
    Implements:
    - Dual loss: HuberLoss + FocalLoss
    - AdamW optimizer with weight decay
    - OneCycleLR scheduler
    - Gradient clipping
    - Training/validation loop
    - Checkpointing
    
    Example:
        >>> config = V4Config()
        >>> model = GoldenBreezeFusionV4(config)
        >>> trainer = TrainerV4(model, config)
        >>> trainer.train(train_loader, val_loader, epochs=50)
    """
    
    def __init__(
        self,
        model: GoldenBreezeFusionV4,
        config: V4Config,
        device: str = None,
        # Loss parameters
        score_weight: float = 1.0,
        class_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Optimizer parameters
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        # Scheduler parameters
        max_lr: float = 1e-3,
        pct_start: float = 0.3,
        # Training parameters
        grad_clip_norm: float = 1.0,
        # Logging
        log_dir: str = "logs/v4_training",
        checkpoint_dir: str = "models",
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.config = config
        
        # Loss parameters
        self.score_weight = score_weight
        self.class_weight = class_weight
        
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.grad_clip_norm = grad_clip_norm
        
        # Directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (classification only for now)
        self.criterion = ClassificationOnlyLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        ).to(self.device)
        
        # Optimizer (will be initialized in train())
        self.optimizer = None
        self.scheduler = None
        
        # Tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_mcc": [],
            "learning_rate": [],
        }
        self.best_val_loss = float("inf")
        self.best_val_mcc = -1.0
        self.epoch = 0
    
    def _init_optimizer_scheduler(self, train_loader: DataLoader, epochs: int):
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        total_steps = len(train_loader) * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            anneal_strategy='cos',
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            x_fast = batch['x_fast'].to(self.device)
            x_slow = batch['x_slow'].to(self.device)
            smc_static = batch['smc_static'].to(self.device)
            smc_dynamic = batch['smc_dynamic'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Filter out invalid labels (-1)
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            x_fast = x_fast[valid_mask]
            x_slow = x_slow[valid_mask]
            smc_static = smc_static[valid_mask]
            smc_dynamic = smc_dynamic[valid_mask]
            labels = labels[valid_mask]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                x_fast_ohlcv=x_fast,
                x_slow_ohlcv=x_slow,
                smc_static=smc_static,
                smc_dynamic=smc_dynamic,
            )
            
            # Compute loss
            loss = self.criterion(output['class_logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.grad_clip_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item() * len(labels)
            total_correct += (output['predicted_class'] == labels).sum().item()
            total_samples += len(labels)
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "samples": total_samples,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict with validation metrics (loss, accuracy, mcc)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in val_loader:
            # Move to device
            x_fast = batch['x_fast'].to(self.device)
            x_slow = batch['x_slow'].to(self.device)
            smc_static = batch['smc_static'].to(self.device)
            smc_dynamic = batch['smc_dynamic'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Filter out invalid labels
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            x_fast = x_fast[valid_mask]
            x_slow = x_slow[valid_mask]
            smc_static = smc_static[valid_mask]
            smc_dynamic = smc_dynamic[valid_mask]
            labels = labels[valid_mask]
            
            # Forward pass
            output = self.model(
                x_fast_ohlcv=x_fast,
                x_slow_ohlcv=x_slow,
                smc_static=smc_static,
                smc_dynamic=smc_dynamic,
            )
            
            # Compute loss
            loss = self.criterion(output['class_logits'], labels)
            total_loss += loss.item() * len(labels)
            
            # Collect predictions
            all_preds.extend(output['predicted_class'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        avg_loss = total_loss / max(len(all_labels), 1)
        accuracy = (all_preds == all_labels).mean() if len(all_labels) > 0 else 0.0
        mcc = self._compute_mcc(all_labels, all_preds)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "mcc": mcc,
            "samples": len(all_labels),
        }
    
    def _compute_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient."""
        try:
            from sklearn.metrics import matthews_corrcoef
            return matthews_corrcoef(y_true, y_pred)
        except ImportError:
            # Simple MCC for binary classification
            if len(np.unique(y_true)) <= 2:
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                
                numerator = (tp * tn) - (fp * fn)
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                
                if denominator == 0:
                    return 0.0
                return numerator / denominator
            else:
                # Multi-class: use accuracy as fallback
                return (y_true == y_pred).mean() * 2 - 1
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            save_best: Save best model checkpoint
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        # Initialize optimizer and scheduler
        self._init_optimizer_scheduler(train_loader, epochs)
        
        # Training loop
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training GoldenBreezeFusionV4")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Learning rate: {self.lr} -> {self.max_lr}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_mcc"].append(val_metrics["mcc"])
            self.history["learning_rate"].append(current_lr)
            
            # Check for improvement
            improved = False
            if val_metrics["mcc"] > self.best_val_mcc:
                self.best_val_mcc = val_metrics["mcc"]
                improved = True
                patience_counter = 0
                
                if save_best:
                    self._save_checkpoint("best_mcc")
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                if save_best and not improved:
                    self._save_checkpoint("best_loss")
            else:
                if not improved:
                    patience_counter += 1
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                status = "★" if improved else " "
                print(
                    f"Epoch {self.epoch:3d}/{epochs} {status} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Val MCC: {val_metrics['mcc']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {self.epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best Val MCC: {self.best_val_mcc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        # Save final model
        self._save_checkpoint("final")
        self._save_history()
        
        return self.history
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"direction_transformer_v4_{name}_{timestamp}.pt"
        path = self.checkpoint_dir / filename
        
        self.model.save(str(path))
        print(f"  Saved checkpoint: {filename}")
    
    def _save_history(self):
        """Save training history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.log_dir / f"training_history_{timestamp}.json"
        
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


def create_dummy_dataset(config: V4Config, n_samples: int = 1000):
    """Create dummy dataset for testing."""
    from aimodule.data_pipeline.alignment import AlignedSample
    
    np.random.seed(42)
    samples = []
    
    for i in range(n_samples):
        sample = AlignedSample(
            timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i),
            m5_window=np.random.randn(config.seq_len_fast, config.input_channels).astype(np.float32),
            h1_window=np.random.randn(config.seq_len_slow, config.input_channels).astype(np.float32),
            smc_static=np.random.randn(config.static_smc_dim).astype(np.float32),
            smc_dynamic=np.random.randn(config.max_dynamic_tokens, config.dynamic_smc_dim).astype(np.float32),
            label=np.random.randint(0, config.num_classes),
        )
        samples.append(sample)
    
    return FusionDataset(samples=samples, config=config)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GoldenBreezeFusionV4")
    
    # Data paths
    parser.add_argument("--m5-path", type=str, default=None, help="Path to M5 CSV")
    parser.add_argument("--h1-path", type=str, default=None, help="Path to H1 CSV")
    parser.add_argument("--labels-path", type=str, default=None, help="Path to labels CSV")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Max learning rate for OneCycleLR")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    
    # Model parameters
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    
    # Other
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--dummy", action="store_true", help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    # Create config
    config = V4Config(
        d_model=args.d_model,
        nhead=args.nhead,
    )
    
    print(f"\n{'='*60}")
    print(f"Golden Breeze Fusion Transformer v4.0 - Training")
    print(f"{'='*60}")
    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  nhead: {config.nhead}")
    print(f"  seq_len_fast: {config.seq_len_fast}")
    print(f"  seq_len_slow: {config.seq_len_slow}")
    
    # Create model
    model = GoldenBreezeFusionV4(config)
    params = model.count_parameters()
    print(f"\nModel Parameters: {params['total']:,}")
    
    # Create dataset
    if args.dummy or (args.m5_path is None):
        print("\n⚠️  Using dummy data for testing...")
        dataset = create_dummy_dataset(config, n_samples=1000)
    else:
        print(f"\nLoading data...")
        print(f"  M5: {args.m5_path}")
        print(f"  H1: {args.h1_path}")
        print(f"  Labels: {args.labels_path}")
        
        # Load CSVs
        df_m5 = pd.read_csv(args.m5_path)
        df_h1 = pd.read_csv(args.h1_path)
        df_labels = pd.read_csv(args.labels_path) if args.labels_path else None
        
        # Create dataset using FusionDatasetV2
        dataset = FusionDatasetV2.from_dataframes(
            df_m5=df_m5,
            df_h1=df_h1,
            df_labels=df_labels,
            config=config,
            label_col='direction_label',  # Column name in labels file
        )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = TrainerV4(
        model=model,
        config=config,
        device=args.device,
        lr=args.lr,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=10,
        save_best=True,
    )
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MCC: {test_metrics['mcc']:.4f}")
    
    return history


if __name__ == "__main__":
    main()

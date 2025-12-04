"""
Golden Breeze v4 - Fusion Dataset for Transformer Training

PyTorch Dataset that provides aligned M5+H1+SMC samples
for the GoldenBreezeFusionV4 model.

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aimodule.data_pipeline.smc_processor import SMCProcessor
from aimodule.data_pipeline.alignment import TimeAligner, AlignedSample
from aimodule.models.v4_transformer.config import V4Config


class FusionDataset(Dataset):
    """
    PyTorch Dataset for Golden Breeze Fusion Transformer v4.
    
    Provides aligned samples of:
    - x_fast: M5 OHLCV sequence (seq_len_fast, 6)
    - x_slow: H1 OHLCV sequence (seq_len_slow, 6)
    - smc_static: Static SMC features (static_smc_dim,)
    - smc_dynamic: Dynamic OB tokens (max_dynamic_tokens, dynamic_smc_dim)
    - label: Direction class (0=DOWN, 1=HOLD, 2=UP)
    
    Example:
        >>> dataset = FusionDataset.from_csv(
        ...     m5_path="data/raw/XAUUSD/M5.csv",
        ...     h1_path="data/raw/XAUUSD/H1.csv",
        ...     labels_path="data/labels/direction_labels.csv",
        ...     config=V4Config(),
        ... )
        >>> sample = dataset[0]
        >>> print(sample['x_fast'].shape)  # (200, 6)
    """
    
    def __init__(
        self,
        samples: List[AlignedSample],
        config: V4Config,
        normalize: bool = True,
    ):
        """
        Args:
            samples: List of AlignedSample objects
            config: V4Config for dimensions
            normalize: Whether to normalize features (already done in alignment)
        """
        self.samples = samples
        self.config = config
        self.normalize = normalize
        
        # Validate dimensions
        if len(samples) > 0:
            sample = samples[0]
            assert sample.m5_window.shape[1] >= 5, "M5 must have at least 5 features"
            assert sample.h1_window.shape[1] >= 5, "H1 must have at least 5 features"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert to tensors
        x_fast = torch.tensor(sample.m5_window, dtype=torch.float32)
        x_slow = torch.tensor(sample.h1_window, dtype=torch.float32)
        smc_static = torch.tensor(sample.smc_static, dtype=torch.float32)
        smc_dynamic = torch.tensor(sample.smc_dynamic, dtype=torch.float32)
        
        # Pad/truncate to expected dimensions
        x_fast = self._pad_sequence(x_fast, self.config.seq_len_fast, self.config.input_channels)
        x_slow = self._pad_sequence(x_slow, self.config.seq_len_slow, self.config.input_channels)
        smc_static = self._pad_vector(smc_static, self.config.static_smc_dim)
        smc_dynamic = self._pad_matrix(
            smc_dynamic, 
            self.config.max_dynamic_tokens, 
            self.config.dynamic_smc_dim
        )
        
        result = {
            'x_fast': x_fast,
            'x_slow': x_slow,
            'smc_static': smc_static,
            'smc_dynamic': smc_dynamic,
        }
        
        if sample.label is not None:
            result['label'] = torch.tensor(sample.label, dtype=torch.long)
        
        return result
    
    def _pad_sequence(
        self, 
        seq: torch.Tensor, 
        target_len: int, 
        target_features: int
    ) -> torch.Tensor:
        """Pad or truncate sequence to target length and features."""
        current_len, current_features = seq.shape
        
        # Handle features
        if current_features < target_features:
            padding = torch.zeros(current_len, target_features - current_features)
            seq = torch.cat([seq, padding], dim=1)
        elif current_features > target_features:
            seq = seq[:, :target_features]
        
        # Handle length
        if current_len < target_len:
            padding = torch.zeros(target_len - current_len, target_features)
            seq = torch.cat([padding, seq], dim=0)  # Pad at beginning
        elif current_len > target_len:
            seq = seq[-target_len:]  # Take last target_len
        
        return seq
    
    def _pad_vector(self, vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Pad vector to target dimension."""
        current_dim = vec.shape[0]
        
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim)
            vec = torch.cat([vec, padding])
        elif current_dim > target_dim:
            vec = vec[:target_dim]
        
        return vec
    
    def _pad_matrix(
        self, 
        mat: torch.Tensor, 
        target_rows: int, 
        target_cols: int
    ) -> torch.Tensor:
        """Pad matrix to target dimensions."""
        current_rows, current_cols = mat.shape
        
        # Handle columns
        if current_cols < target_cols:
            padding = torch.zeros(current_rows, target_cols - current_cols)
            mat = torch.cat([mat, padding], dim=1)
        elif current_cols > target_cols:
            mat = mat[:, :target_cols]
        
        # Handle rows
        current_rows = mat.shape[0]
        if current_rows < target_rows:
            padding = torch.zeros(target_rows - current_rows, target_cols)
            mat = torch.cat([mat, padding], dim=0)
        elif current_rows > target_rows:
            mat = mat[:target_rows]
        
        return mat
    
    @classmethod
    def from_csv(
        cls,
        m5_path: str,
        h1_path: str,
        labels_path: str,
        config: Optional[V4Config] = None,
        label_col: str = 'label',
        time_col: str = 'time',
    ) -> 'FusionDataset':
        """
        Create dataset from CSV files.
        
        Args:
            m5_path: Path to M5 OHLCV CSV
            h1_path: Path to H1 OHLCV CSV
            labels_path: Path to labels CSV
            config: V4Config (default if None)
            label_col: Label column name
            time_col: Time column name
        """
        config = config or V4Config()
        
        # Load data
        print(f"Loading M5 data from {m5_path}...")
        df_m5 = pd.read_csv(m5_path)
        df_m5[time_col] = pd.to_datetime(df_m5[time_col])
        
        print(f"Loading H1 data from {h1_path}...")
        df_h1 = pd.read_csv(h1_path)
        df_h1[time_col] = pd.to_datetime(df_h1[time_col])
        
        print(f"Loading labels from {labels_path}...")
        df_labels = pd.read_csv(labels_path)
        df_labels[time_col] = pd.to_datetime(df_labels[time_col])
        
        print(f"M5 bars: {len(df_m5)}, H1 bars: {len(df_h1)}, Labels: {len(df_labels)}")
        
        # Calculate SMC features
        print("Calculating SMC features...")
        smc_processor = SMCProcessor(
            decay_lambda=config.smc_decay_lambda,
            max_ob_age=config.smc_max_ob_age,
            max_active_obs=config.max_dynamic_tokens,
        )
        df_smc = smc_processor.calculate_ob_features(df_h1)
        print(f"SMC features calculated: {len(df_smc)} H1 bars")
        
        # Create aligned samples
        print("Creating aligned samples...")
        aligner = TimeAligner()
        samples = aligner.create_training_samples(
            df_m5=df_m5,
            df_h1=df_h1,
            df_smc=df_smc,
            df_labels=df_labels,
            seq_len_fast=config.seq_len_fast,
            seq_len_slow=config.seq_len_slow,
            label_col=label_col,
        )
        
        print(f"Created {len(samples)} aligned samples")
        
        return cls(samples=samples, config=config)
    
    @classmethod
    def from_npz(cls, npz_path: str, config: Optional[V4Config] = None) -> 'FusionDataset':
        """
        Load dataset from NPZ file.
        
        Args:
            npz_path: Path to NPZ file
            config: V4Config
        """
        config = config or V4Config()
        
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        
        samples = []
        n_samples = len(data['m5_windows'])
        
        for i in range(n_samples):
            sample = AlignedSample(
                timestamp=pd.Timestamp(data['timestamps'][i]),
                m5_window=data['m5_windows'][i],
                h1_window=data['h1_windows'][i],
                smc_static=data['smc_static'][i],
                smc_dynamic=data['smc_dynamic'][i],
                label=int(data['labels'][i]) if 'labels' in data else None,
            )
            samples.append(sample)
        
        print(f"Loaded {len(samples)} samples")
        
        return cls(samples=samples, config=config)
    
    def save_npz(self, npz_path: str):
        """Save dataset to NPZ file."""
        timestamps = [s.timestamp.isoformat() for s in self.samples]
        m5_windows = np.stack([s.m5_window for s in self.samples])
        h1_windows = np.stack([s.h1_window for s in self.samples])
        smc_static = np.stack([s.smc_static for s in self.samples])
        smc_dynamic = np.stack([s.smc_dynamic for s in self.samples])
        labels = np.array([s.label if s.label is not None else -1 for s in self.samples])
        
        np.savez_compressed(
            npz_path,
            timestamps=timestamps,
            m5_windows=m5_windows,
            h1_windows=h1_windows,
            smc_static=smc_static,
            smc_dynamic=smc_dynamic,
            labels=labels,
        )
        
        print(f"Saved {len(self.samples)} samples to {npz_path}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = [s.label for s in self.samples if s.label is not None]
        
        if not labels:
            return torch.ones(self.config.num_classes)
        
        counts = np.bincount(labels, minlength=self.config.num_classes)
        weights = 1.0 / (counts + 1)  # Add 1 to avoid division by zero
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_splits(
        self, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Tuple['FusionDataset', 'FusionDataset', 'FusionDataset']:
        """
        Split dataset into train/val/test.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (test = 1 - train - val)
            shuffle: Whether to shuffle before splitting
            seed: Random seed
        """
        n = len(self.samples)
        indices = list(range(n))
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_samples = [self.samples[i] for i in indices[:train_end]]
        val_samples = [self.samples[i] for i in indices[train_end:val_end]]
        test_samples = [self.samples[i] for i in indices[val_end:]]
        
        return (
            FusionDataset(train_samples, self.config),
            FusionDataset(val_samples, self.config),
            FusionDataset(test_samples, self.config),
        )


def create_dataloaders(
    dataset: FusionDataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from FusionDataset.
    
    Args:
        dataset: FusionDataset instance
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        num_workers: DataLoader workers
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = dataset.get_splits(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle=False,  # Don't shuffle for time series
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("FusionDataset - Quick Test")
    print("=" * 60)
    
    # Create dummy samples
    np.random.seed(42)
    config = V4Config()
    
    samples = []
    for i in range(100):
        sample = AlignedSample(
            timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i),
            m5_window=np.random.randn(config.seq_len_fast, 5).astype(np.float32),
            h1_window=np.random.randn(config.seq_len_slow, 5).astype(np.float32),
            smc_static=np.random.randn(config.static_smc_dim).astype(np.float32),
            smc_dynamic=np.random.randn(config.max_dynamic_tokens, config.dynamic_smc_dim).astype(np.float32),
            label=np.random.randint(0, 3),
        )
        samples.append(sample)
    
    dataset = FusionDataset(samples=samples, config=config)
    print(f"Dataset size: {len(dataset)}")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  x_fast: {sample['x_fast'].shape}")
    print(f"  x_slow: {sample['x_slow'].shape}")
    print(f"  smc_static: {sample['smc_static'].shape}")
    print(f"  smc_dynamic: {sample['smc_dynamic'].shape}")
    print(f"  label: {sample['label']}")
    
    # Test splits
    train_ds, val_ds, test_ds = dataset.get_splits()
    print(f"\nSplits:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")
    print(f"  Test: {len(test_ds)}")
    
    # Test DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=16,
    )
    
    print(f"\nDataLoader test:")
    for batch in train_loader:
        print(f"  Batch x_fast: {batch['x_fast'].shape}")
        print(f"  Batch x_slow: {batch['x_slow'].shape}")
        print(f"  Batch smc_static: {batch['smc_static'].shape}")
        print(f"  Batch smc_dynamic: {batch['smc_dynamic'].shape}")
        print(f"  Batch labels: {batch['label'].shape}")
        break
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"\nClass weights: {weights}")
    
    print("\n✅ FusionDataset test passed!")

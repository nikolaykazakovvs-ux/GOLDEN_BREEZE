"""
Evaluate the best trained BTC v5 model on test set.
Uses the existing best_model.pt checkpoint.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aimodule.models.v5_ultimate import GoldenBreezeV5Ultimate


class BTCTestDataset(Dataset):
    """Dataset for evaluation (uses mmap for efficient access)."""
    
    CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']
    
    def __init__(self, npz_path: str, indices: np.ndarray = None):
        # Load with mmap
        self.data_dict = np.load(npz_path, mmap_mode='r', allow_pickle=False)
        
        self.x_fast = self.data_dict['x_fast']
        self.x_slow = self.data_dict['x_slow']
        self.x_strategy = self.data_dict['x_strategy']
        labels_5class = self.data_dict['y']
        
        # Map 5-class to 3-class
        labels_3class = np.where(
            labels_5class <= 1, 0,
            np.where(labels_5class == 2, 1, 2)
        )
        
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(labels_3class))
        
        self.labels = torch.from_numpy(labels_3class[self.indices]).long()
        self.n_samples = len(self.labels)
        
        print(f"   ✅ Loaded {self.n_samples:,} samples (using mmap)")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            torch.from_numpy(self.x_fast[real_idx]).float(),
            torch.from_numpy(self.x_slow[real_idx]).float(),
            torch.from_numpy(self.x_strategy[real_idx]).float(),
            self.labels[idx]
        )


@torch.no_grad()
def evaluate_model(model, loader, device, class_names):
    """Evaluate model on loader."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for x_fast, x_slow, x_strat, labels in pbar:
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
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mcc': mcc,
        'preds': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'models/v5_btc/best_model.pt'
    data_path = 'data/prepared/btc_v5.npz'
    
    print("=" * 70)
    print("🪙 Golden Breeze v5 BTC - Model Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    # Load checkpoint
    print("\n📂 Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    best_epoch = checkpoint.get('epoch', '?')
    best_mcc = checkpoint.get('val_mcc', '?')
    
    print(f"   Epoch: {best_epoch}")
    print(f"   Best Val MCC: {best_mcc}")
    
    # Create model
    from aimodule.models.v5_ultimate import V5UltimateConfig
    model_config = V5UltimateConfig()
    model = GoldenBreezeV5Ultimate(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = model.count_parameters()
    print(f"   Parameters: {total_params:,}")
    
    # Load data splits
    print("\n📂 Creating data splits...")
    data = np.load(data_path)
    n_samples = len(data['y'])
    
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    test_idx = indices[train_size + val_size:]
    
    print(f"   Test samples: {len(test_idx):,}")
    
    # Create dataset and loader
    print("\n📂 Loading test dataset...")
    test_ds = BTCTestDataset(data_path, test_idx)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )
    
    # Evaluate
    print("\n🚀 Evaluating on test set...")
    results = evaluate_model(model, test_loader, device, ['DOWN', 'NEUTRAL', 'UP'])
    
    print("\n" + "=" * 70)
    print("📈 Test Results")
    print("=" * 70)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"MCC: {results['mcc']:+.4f}")
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print(f"\nPer-Class Accuracy:")
    cm = results['confusion_matrix']
    for i, name in enumerate(['DOWN', 'NEUTRAL', 'UP']):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"   {name}: {acc:.4f}")
    
    # Save report
    report = {
        'model': 'GoldenBreezeV5Ultimate_BTC',
        'checkpoint_epoch': int(best_epoch) if best_epoch != '?' else None,
        'checkpoint_val_mcc': float(best_mcc) if best_mcc != '?' else None,
        'test_metrics': {
            'loss': float(results['loss']),
            'accuracy': float(results['accuracy']),
            'mcc': float(results['mcc']),
        },
        'device': device,
        'total_params': total_params,
    }
    
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/v5_btc_evaluation.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Report saved to {report_path}")


if __name__ == "__main__":
    main()

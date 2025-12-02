"""
Train Direction LSTM model from prepared dataset.

Uses existing Direction LSTM architecture from aimodule.models.direction_lstm_model.

Usage:
    python -m aimodule.training.train_direction_lstm_from_labels \
        --data data/prepared/direction_dataset.npz \
        --seq-len 50 \
        --epochs 20 \
        --batch-size 64 \
        --lr 1e-3 \
        --save-path models/direction_lstm_hybrid.pt

Author: Golden Breeze Team
Version: 1.1
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Fix seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DirectionDataset(Dataset):
    """PyTorch Dataset for Direction LSTM."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y - 1)  # Convert {1,2} to {0,1} for CrossEntropyLoss
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DirectionLSTM(nn.Module):
    """
    Direction LSTM model (using existing architecture).
    
    Architecture:
    - LSTM: 2 layers, 64 hidden units
    - Dropout: 0.3
    - FC: hidden -> num_classes
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 num_classes: int = 2, dropout: float = 0.3):
        super(DirectionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Dropout + FC
        out = self.dropout(last_out)
        out = self.fc(out)  # (batch, num_classes)
        
        return out


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Direction LSTM from labels"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to prepared dataset (.npz file)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Sequence length (must match dataset)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden size"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/direction_lstm_hybrid.pt",
        help="Path to save trained model"
    )
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = y_batch.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return avg_loss, acc, f1, mcc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return avg_loss, acc, f1, mcc, all_preds, all_labels


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*60)
    print("Golden Breeze - Direction LSTM Training")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.data}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Patience: {args.patience}")
    print(f"Save path: {args.save_path}")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    data = np.load(args.data, allow_pickle=True)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    n_features = data['n_features']
    n_classes = int(data['n_classes'])
    
    print(f"✅ Dataset loaded")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")
    
    # Create datasets
    train_dataset = DirectionDataset(X_train, y_train)
    val_dataset = DirectionDataset(X_val, y_val)
    test_dataset = DirectionDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\nInitializing model...")
    model = DirectionLSTM(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=n_classes,
        dropout=args.dropout
    ).to(DEVICE)
    
    print(f"✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "="*60)
    print("Training started...")
    print("="*60)
    
    best_mcc = -1.0
    patience_counter = 0
    train_history = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_f1, train_mcc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_mcc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        # Log
        print(f"Epoch {epoch:3d}/{args.epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}, MCC={train_mcc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, MCC={val_mcc:.4f}")
        
        # Save history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_mcc': train_mcc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_mcc': val_mcc,
        })
        
        # Early stopping on MCC
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            
            # Save best model
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best model saved (MCC={best_mcc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break
    
    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Loading best model for final evaluation...")
    print("="*60)
    
    model.load_state_dict(torch.load(args.save_path))
    
    # Final test evaluation
    test_loss, test_acc, test_f1, test_mcc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    print(f"\n✅ Final Test Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 (macro): {test_f1:.4f}")
    print(f"   MCC: {test_mcc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save metadata
    meta_path = Path(args.save_path).with_suffix('.json')
    metadata = {
        'model_type': 'DirectionLSTM',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_path': args.data,
        'seq_len': args.seq_len,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'n_features': int(n_features),
        'n_classes': int(n_classes),
        'feature_cols': data['feature_cols'].tolist(),
        'symbol': str(data['symbol']),
        'timeframe': str(data['timeframe']),
        'epochs_trained': len(train_history),
        'best_val_mcc': float(best_mcc),
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1),
            'mcc': float(test_mcc),
            'loss': float(test_loss),
        },
        'confusion_matrix': cm.tolist(),
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to {meta_path}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {args.save_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"\nNext step:")
    print(f"  python -m tools.train_and_backtest_hybrid \\")
    print(f"    --symbol {metadata['symbol']} \\")
    print(f"    --model {args.save_path}")


if __name__ == "__main__":
    main()

"""
Full Training Pipeline: Train models on all key timeframes for XAUUSD 2025
Then test on the last available week
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# ============== CONFIGURATION ==============
SYMBOL = "XAUUSD"
TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4
}

# Training period: Jan 1, 2025 - Nov 24, 2025 (before last week)
# Test period: Nov 25, 2025 - Dec 4, 2025 (last ~week)
TRAIN_END = datetime(2025, 11, 24)
TEST_START = datetime(2025, 11, 25)

SEQ_LEN = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============== MODEL ==============
class DirectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ============== DATA LOADING ==============
def load_mt5_data(symbol, timeframe, start_date, end_date):
    """Load OHLCV data from MT5"""
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df


def calculate_features(df):
    """Calculate technical indicators as features"""
    data = df.copy()
    
    # Price features
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # SMA
    data['sma_fast'] = data['close'].rolling(20).mean()
    data['sma_slow'] = data['close'].rolling(50).mean()
    data['sma_ratio'] = data['sma_fast'] / data['sma_slow']
    
    # ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(14).mean()
    data['atr_norm'] = data['atr'] / data['close']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_norm'] = data['rsi'] / 100
    
    # Bollinger Bands
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - bb_mid) / (2 * bb_std + 1e-10)
    
    # Volume
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    
    # Momentum
    data['momentum'] = data['close'] / data['close'].shift(10) - 1
    data['roc'] = data['close'].pct_change(10)
    
    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    data = data.ffill().bfill()
    return data


def generate_labels(df, horizon=12, threshold=0.001):
    """Generate direction labels: 1=UP, 0=DOWN"""
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    labels = (future_return > threshold).astype(int)
    return labels


def create_sequences(features, labels, seq_len):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(seq_len, len(features) - 1):
        if pd.notna(labels.iloc[i]):
            X.append(features[i-seq_len:i])
            y.append(labels.iloc[i])
    return np.array(X), np.array(y)


# ============== TRAINING ==============
def train_model(X_train, y_train, X_val, y_val, input_size, device):
    """Train LSTM model with early stopping"""
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = DirectionLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, 2, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_mcc = -1
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_y.numpy())
        
        val_mcc = matthews_corrcoef(val_true, val_preds)
        val_acc = accuracy_score(val_true, val_preds)
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val MCC: {val_mcc:.4f} | Val Acc: {val_acc:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_val_mcc


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model on test set"""
    model.eval()
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_true.extend(batch_y.numpy())
            all_probs.extend(probs[:, 1])  # Probability of UP
    
    metrics = {
        'accuracy': accuracy_score(all_true, all_preds),
        'f1': f1_score(all_true, all_preds, average='macro'),
        'mcc': matthews_corrcoef(all_true, all_preds),
        'confusion_matrix': confusion_matrix(all_true, all_preds).tolist()
    }
    
    return metrics, all_preds, all_probs


# ============== MAIN PIPELINE ==============
def main():
    print("="*70)
    print("MULTI-TIMEFRAME TRAINING PIPELINE")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   Symbol: {SYMBOL}")
    print(f"   Timeframes: {list(TIMEFRAMES.keys())}")
    print(f"   Training: Jan 1, 2025 - Nov 24, 2025")
    print(f"   Testing: Nov 25, 2025 - Dec 4, 2025")
    print(f"   Device: {DEVICE}")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return
    print("\n✅ MT5 connected")
    
    results = {}
    
    for tf_name, tf in TIMEFRAMES.items():
        print("\n" + "="*70)
        print(f"TIMEFRAME: {tf_name}")
        print("="*70)
        
        # Load training data
        print(f"\n[1] Loading training data...")
        train_df = load_mt5_data(SYMBOL, tf, datetime(2025, 1, 1), TRAIN_END)
        if train_df is None or len(train_df) < 500:
            print(f"❌ Not enough training data for {tf_name}")
            continue
        print(f"   Training bars: {len(train_df)} ({train_df.index[0]} to {train_df.index[-1]})")
        
        # Load test data
        print(f"\n[2] Loading test data (last week)...")
        test_df = load_mt5_data(SYMBOL, tf, TEST_START, datetime(2025, 12, 31))
        if test_df is None or len(test_df) < 50:
            print(f"❌ Not enough test data for {tf_name}")
            continue
        print(f"   Test bars: {len(test_df)} ({test_df.index[0]} to {test_df.index[-1]})")
        
        # Calculate features
        print(f"\n[3] Calculating features...")
        train_features = calculate_features(train_df)
        test_features = calculate_features(test_df)
        
        # Generate labels
        print(f"\n[4] Generating labels...")
        train_labels = generate_labels(train_features, horizon=12, threshold=0.001)
        test_labels = generate_labels(test_features, horizon=12, threshold=0.001)
        
        # Feature columns
        feature_cols = ['close', 'returns', 'log_returns', 'sma_fast', 'sma_slow', 'sma_ratio',
                       'atr', 'atr_norm', 'rsi_norm', 'bb_position', 'volume_ratio',
                       'momentum', 'roc', 'macd', 'macd_signal', 'macd_hist']
        
        train_feat_values = train_features[feature_cols].values
        test_feat_values = test_features[feature_cols].values
        
        # Normalize
        scaler = StandardScaler()
        train_feat_norm = scaler.fit_transform(train_feat_values)
        test_feat_norm = scaler.transform(test_feat_values)
        
        # Create sequences
        print(f"\n[5] Creating sequences...")
        X_train, y_train = create_sequences(train_feat_norm, train_labels, SEQ_LEN)
        X_test, y_test = create_sequences(test_feat_norm, test_labels, SEQ_LEN)
        
        print(f"   Train sequences: {len(X_train)}")
        print(f"   Test sequences: {len(X_test)}")
        
        if len(X_train) < 100:
            print(f"❌ Not enough sequences for {tf_name}")
            continue
        
        # Split train/val
        val_size = int(len(X_train) * 0.15)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        
        print(f"   Train/Val split: {len(X_train)}/{len(X_val)}")
        
        # Train model
        print(f"\n[6] Training model on GPU...")
        import time
        start_time = time.time()
        model, best_val_mcc = train_model(X_train, y_train, X_val, y_val, len(feature_cols), DEVICE)
        train_time = time.time() - start_time
        print(f"   Training time: {train_time:.1f}s")
        print(f"   Best Val MCC: {best_val_mcc:.4f}")
        
        # Evaluate on test set (last week)
        print(f"\n[7] Evaluating on last week...")
        test_metrics, test_preds, test_probs = evaluate_model(model, X_test, y_test, DEVICE)
        
        print(f"\n📈 Test Results ({tf_name}):")
        print(f"   Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"   F1 Score: {test_metrics['f1']*100:.2f}%")
        print(f"   MCC: {test_metrics['mcc']:.4f}")
        print(f"   Confusion Matrix:")
        cm = test_metrics['confusion_matrix']
        print(f"      DOWN: {cm[0][0]} correct, {cm[0][1]} wrong")
        print(f"      UP:   {cm[1][0]} wrong, {cm[1][1]} correct")
        
        # Save model
        model_path = f"models/direction_lstm_{tf_name}_2025.pt"
        torch.save(model.state_dict(), model_path)
        print(f"\n   ✅ Model saved: {model_path}")
        
        # Store results
        results[tf_name] = {
            'train_bars': len(train_df),
            'test_bars': len(test_df),
            'train_sequences': len(X_train) + len(X_val),
            'test_sequences': len(X_test),
            'train_time': train_time,
            'val_mcc': best_val_mcc,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_mcc': test_metrics['mcc'],
            'confusion_matrix': cm
        }
    
    mt5.shutdown()
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: All Timeframes")
    print("="*70)
    
    print("\n📊 Training Results:")
    print("-"*80)
    print(f"{'TF':<6} | {'Train Bars':>10} | {'Sequences':>10} | {'Time':>8} | {'Val MCC':>8} | {'Test MCC':>8} | {'Test Acc':>8}")
    print("-"*80)
    
    for tf_name, r in results.items():
        print(f"{tf_name:<6} | {r['train_bars']:>10} | {r['train_sequences']:>10} | {r['train_time']:>7.1f}s | {r['val_mcc']:>8.4f} | {r['test_mcc']:>8.4f} | {r['test_accuracy']*100:>7.2f}%")
    print("-"*80)
    
    # Best model
    if results:
        best_tf = max(results.keys(), key=lambda x: results[x]['test_mcc'])
        print(f"\n🏆 Best Model: {best_tf} (Test MCC: {results[best_tf]['test_mcc']:.4f})")
    
    # Save results
    with open('models/training_results_2025.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to models/training_results_2025.json")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()

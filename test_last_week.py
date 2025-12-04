"""
Test Model v3 on Last Week (Nov 25 - Dec 4, 2025)
Compare with baseline and run backtest
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import MetaTrader5 as mt5

# ============== MODEL ==============
class DirectionLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    confidence: float

def load_model_v3():
    """Load trained model v3"""
    model_path = "models/direction_lstm_gold_v3.pt"
    meta_path = "models/direction_lstm_gold_v3.json"
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    n_features = metadata.get('n_features', 32)
    model = DirectionLSTM(
        input_size=n_features,
        hidden_size=metadata.get('hidden_size', 64),
        num_layers=metadata.get('num_layers', 2),
        num_classes=metadata.get('n_classes', 2),
        dropout=metadata.get('dropout', 0.3)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, metadata

def calculate_indicators(df):
    """Add technical indicators"""
    data = df.copy()
    
    # SMA
    data['sma_fast'] = data['close'].rolling(20).mean()
    data['sma_slow'] = data['close'].rolling(50).mean()
    
    # ATR
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(14).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger position
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - bb_mid) / (2 * bb_std + 1e-10)
    
    data = data.ffill().bfill()
    return data

def prepare_features(row, prev_close):
    """Prepare 32-feature vector"""
    close = row['close']
    sma_fast = row.get('sma_fast', close)
    sma_slow = row.get('sma_slow', close)
    atr = row.get('atr', 0)
    rsi = row.get('rsi', 50)
    
    returns = (close - prev_close) / prev_close if prev_close else 0
    log_returns = np.log(close / prev_close) if prev_close and prev_close > 0 else 0
    
    feature_vec = [
        close,
        returns,
        log_returns,
        sma_fast if not np.isnan(sma_fast) else close,
        sma_slow if not np.isnan(sma_slow) else close,
        sma_fast / sma_slow if sma_slow else 1.0,
        atr if not np.isnan(atr) else 0,
        atr / close if atr and close else 0,
        rsi / 100 if not np.isnan(rsi) else 0.5,
        row.get('bb_position', 0),
        0, 0, 0, 0, 0, 0, 0, 0,  # SMC features placeholder
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Gold features placeholder
    ]
    
    while len(feature_vec) < 32:
        feature_vec.append(0.0)
    
    return feature_vec[:32]

def run_backtest(data, model, device, config):
    """Run backtest on data"""
    seq_len = 50
    min_confidence = config['min_confidence']
    atr_sl_mult = config['atr_sl_mult']
    atr_tp_mult = config['atr_tp_mult']
    cooldown = config['cooldown']
    spread = config['spread']
    initial_balance = config['initial_balance']
    
    model = model.to(device)
    model.eval()
    
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0.0
    
    trades: List[Trade] = []
    feature_buffer = []
    current_trade: Optional[Trade] = None
    bars_since_trade = cooldown
    
    warmup = max(seq_len, 50)
    
    for i in range(warmup, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        timestamp = data.index[i]
        price = row['close']
        atr = row['atr'] if not np.isnan(row['atr']) else 10
        
        # Update features
        features = prepare_features(row, prev_row['close'])
        feature_buffer.append(features)
        if len(feature_buffer) > seq_len:
            feature_buffer = feature_buffer[-seq_len:]
        
        # Check open trade
        if current_trade:
            if current_trade.direction == 'long':
                sl = current_trade.entry_price - atr * atr_sl_mult
                tp = current_trade.entry_price + atr * atr_tp_mult
                if row['low'] <= sl:
                    current_trade.exit_price = sl
                    current_trade.pnl = (sl - current_trade.entry_price) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
                elif row['high'] >= tp:
                    current_trade.exit_price = tp
                    current_trade.pnl = (tp - current_trade.entry_price) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
            else:
                sl = current_trade.entry_price + atr * atr_sl_mult
                tp = current_trade.entry_price - atr * atr_tp_mult
                if row['high'] >= sl:
                    current_trade.exit_price = sl
                    current_trade.pnl = (current_trade.entry_price - sl) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
                elif row['low'] <= tp:
                    current_trade.exit_price = tp
                    current_trade.pnl = (current_trade.entry_price - tp) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
            continue
        
        bars_since_trade += 1
        
        if len(feature_buffer) < seq_len:
            continue
        
        # Predict
        seq = np.array(feature_buffer[-seq_len:])
        seq_norm = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
        
        with torch.no_grad():
            x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        if confidence >= min_confidence and bars_since_trade >= cooldown:
            direction = 'long' if pred_class == 1 else 'short'
            entry_price = price + spread if direction == 'long' else price - spread
            
            current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                direction=direction,
                entry_price=entry_price,
                exit_price=None,
                pnl=None,
                confidence=confidence
            )
            bars_since_trade = 0
        
        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Close open trade
    if current_trade:
        current_trade.exit_price = data.iloc[-1]['close']
        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * 100 if current_trade.direction == 'long' else (current_trade.entry_price - current_trade.exit_price) * 100
        current_trade.exit_time = data.index[-1]
        trades.append(current_trade)
        balance += current_trade.pnl
    
    return trades, balance, max_drawdown

def main():
    print("="*70)
    print("MODEL V3 - TEST ON LAST WEEK (Nov 25 - Dec 4, 2025)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 failed")
        return
    print("✅ MT5 connected")
    
    # Load model v3
    print("\n[1] Loading Model v3...")
    model, metadata = load_model_v3()
    print(f"✅ Model loaded: {metadata.get('n_features', 32)} features, MCC={metadata['test_metrics']['mcc']:.4f}")
    
    # Test on different timeframes using same model
    timeframes = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1
    }
    
    # Backtest configs for different TFs
    configs = {
        "M5": {
            'min_confidence': 0.95,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 3.0,
            'cooldown': 24,
            'spread': 0.30,
            'initial_balance': 10000.0
        },
        "M15": {
            'min_confidence': 0.90,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.5,
            'cooldown': 8,
            'spread': 0.30,
            'initial_balance': 10000.0
        },
        "H1": {
            'min_confidence': 0.85,
            'atr_sl_mult': 1.5,
            'atr_tp_mult': 2.0,
            'cooldown': 2,
            'spread': 0.30,
            'initial_balance': 10000.0
        }
    }
    
    results = {}
    
    for tf_name, tf in timeframes.items():
        print(f"\n{'='*70}")
        print(f"TESTING ON {tf_name} - LAST WEEK")
        print("="*70)
        
        # Load data
        print(f"\n[2] Loading {tf_name} data...")
        rates = mt5.copy_rates_range("XAUUSD", tf, datetime(2025, 11, 25), datetime(2025, 12, 31))
        if rates is None or len(rates) < 100:
            print(f"❌ Not enough data for {tf_name}")
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        print(f"   Bars: {len(df)}")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        
        # Calculate indicators
        print(f"\n[3] Calculating indicators...")
        data = calculate_indicators(df)
        print("✅ Indicators ready")
        
        # Run backtest
        print(f"\n[4] Running backtest...")
        config = configs[tf_name]
        trades, final_balance, max_dd = run_backtest(data, model, device, config)
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')
        roi = (total_pnl / config['initial_balance']) * 100
        
        print(f"\n📈 Results ({tf_name}):")
        print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Total Bars: {len(df)}")
        print(f"   Total Trades: {len(trades)}")
        print(f"   Wins/Losses: {len(wins)}/{len(losses)}")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        print(f"   ROI: {roi:.2f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Max Drawdown: {max_dd:.2f}%")
        
        results[tf_name] = {
            'bars': len(df),
            'period_start': df.index[0].strftime('%Y-%m-%d %H:%M'),
            'period_end': df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    mt5.shutdown()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Model v3 Performance on Last Week")
    print("="*70)
    
    print("\n📊 Results by Timeframe:")
    print("-"*90)
    print(f"{'TF':<6} | {'Period':>25} | {'Trades':>7} | {'Win%':>6} | {'P&L':>12} | {'ROI':>8} | {'PF':>6}")
    print("-"*90)
    
    for tf_name, r in results.items():
        period = f"{r['period_start'][:10]} - {r['period_end'][:10]}"
        pnl_str = f"${r['total_pnl']:+,.2f}"
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
        print(f"{tf_name:<6} | {period:>25} | {r['trades']:>7} | {r['win_rate']:>5.1f}% | {pnl_str:>12} | {r['roi']:>7.1f}% | {pf_str:>6}")
    print("-"*90)
    
    # Best TF
    if results:
        best_tf = max(results.keys(), key=lambda x: results[x]['total_pnl'])
        worst_tf = min(results.keys(), key=lambda x: results[x]['total_pnl'])
        print(f"\n🏆 Best Performance: {best_tf} (P&L: ${results[best_tf]['total_pnl']:+,.2f}, ROI: {results[best_tf]['roi']:.1f}%)")
        print(f"⚠️  Worst Performance: {worst_tf} (P&L: ${results[worst_tf]['total_pnl']:+,.2f}, ROI: {results[worst_tf]['roi']:.1f}%)")
    
    # Save results
    with open('models/last_week_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to models/last_week_test_results.json")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

if __name__ == "__main__":
    main()

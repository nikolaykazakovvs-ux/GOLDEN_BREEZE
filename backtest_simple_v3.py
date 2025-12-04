"""
Golden Breeze - Simplified Backtest v3 with Direct Trade Execution
Uses real LSTM Direction model v3 for prediction
Simulates trades directly without pending orders
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

# LSTM Model Architecture (same as training)
class DirectionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@dataclass 
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    pnl: Optional[float]
    regime: str
    confidence: float

def load_model():
    """Load LSTM v3 model"""
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # Bollinger Bands position
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - bb_mid) / (2 * bb_std + 1e-10)
    
    data = data.ffill().bfill()
    return data

def prepare_features(row, prev_close):
    """Prepare feature vector for a single bar"""
    close = row['close']
    sma_fast = row.get('sma_fast', close)
    sma_slow = row.get('sma_slow', close)
    atr = row.get('atr', 0)
    rsi = row.get('rsi', 50)
    
    returns = (close - prev_close) / prev_close if prev_close else 0
    log_returns = np.log(close / prev_close) if prev_close and prev_close > 0 else 0
    
    feature_vec = [
        close,                                           # close
        returns,                                          # returns
        log_returns,                                      # log_returns
        sma_fast if not np.isnan(sma_fast) else close,   # sma_fast
        sma_slow if not np.isnan(sma_slow) else close,   # sma_slow
        sma_fast / sma_slow if sma_slow else 1.0,        # sma_ratio
        atr if not np.isnan(atr) else 0,                 # atr
        atr / close if atr and close else 0,             # atr_norm
        rsi / 100 if not np.isnan(rsi) else 0.5,         # rsi
        row.get('bb_position', 0),                       # bb_position
        0, 0, 0, 0, 0, 0, 0, 0,                         # SMC features (12-19)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,             # Gold features (20-31)
    ]
    
    # Ensure exactly 32 features
    while len(feature_vec) < 32:
        feature_vec.append(0.0)
    
    return feature_vec[:32]

def main():
    print("="*60)
    print("Golden Breeze - Simple Backtest v3")
    print("Direct Trade Execution with LSTM Model")
    print("="*60)
    
    # Connect to MT5
    print("\n[1] Connecting to MT5...")
    from mcp_servers.trading.mt5_connector import MT5Connector
    from mcp_servers.trading import market_data
    connector = MT5Connector()
    if not connector.initialize():
        print("❌ MT5 connection failed")
        return
    print("✅ Connected to MT5")
    
    # Load data
    print("\n[2] Loading M5 data...")
    m5_data = market_data.get_ohlcv("XAUUSD", "M5", count=5000)
    if m5_data is None or len(m5_data) < 500:
        print("❌ Not enough data")
        connector.shutdown()
        return
    print(f"✅ Loaded {len(m5_data)} bars")
    print(f"   Period: {m5_data.index[0]} to {m5_data.index[-1]}")
    
    # Calculate indicators
    print("\n[3] Calculating indicators...")
    data = calculate_indicators(m5_data)
    print("✅ Indicators ready")
    
    # Load model
    print("\n[4] Loading LSTM v3 model...")
    model, metadata = load_model()
    print(f"✅ Model loaded: {metadata.get('n_features', 32)} features")
    
    # Backtest parameters
    print("\n[5] Starting backtest...")
    initial_balance = 10000.0
    risk_per_trade = 0.01  # 1%
    min_confidence = 0.95  # Very high threshold - only best signals
    atr_sl_mult = 1.5      # SL = ATR * mult (tighter stop)
    atr_tp_mult = 3.0      # TP = ATR * mult (R:R = 1:2)
    seq_len = 50           # Sequence length for LSTM
    spread = 0.30          # Spread in price points (~30 pips for XAUUSD)
    min_bars_between_trades = 24  # 2 hour cooldown between trades
    
    balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0.0
    
    trades: List[Trade] = []
    feature_buffer = []
    
    current_trade: Optional[Trade] = None
    bars_since_last_trade = min_bars_between_trades  # Allow first trade
    
    # Progress tracking
    total_bars = len(data)
    warmup = max(seq_len, 50)
    
    for i in range(warmup, total_bars):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        timestamp = data.index[i]
        
        price = row['close']
        atr = row['atr'] if not np.isnan(row['atr']) else 10
        
        # Update feature buffer
        features = prepare_features(row, prev_row['close'])
        feature_buffer.append(features)
        if len(feature_buffer) > seq_len:
            feature_buffer = feature_buffer[-seq_len:]
        
        # Check if we have open trade
        if current_trade:
            # Check SL/TP
            if current_trade.direction == 'long':
                sl = current_trade.entry_price - atr * atr_sl_mult
                tp = current_trade.entry_price + atr * atr_tp_mult
                if row['low'] <= sl:
                    # Stop loss hit
                    current_trade.exit_price = sl
                    current_trade.pnl = (sl - current_trade.entry_price) * 100  # pips
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
                elif row['high'] >= tp:
                    # Take profit hit
                    current_trade.exit_price = tp
                    current_trade.pnl = (tp - current_trade.entry_price) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
            else:  # short
                sl = current_trade.entry_price + atr * atr_sl_mult
                tp = current_trade.entry_price - atr * atr_tp_mult
                if row['high'] >= sl:
                    # Stop loss hit
                    current_trade.exit_price = sl
                    current_trade.pnl = (current_trade.entry_price - sl) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
                elif row['low'] <= tp:
                    # Take profit hit
                    current_trade.exit_price = tp
                    current_trade.pnl = (current_trade.entry_price - tp) * 100
                    current_trade.exit_time = timestamp
                    trades.append(current_trade)
                    balance += current_trade.pnl
                    current_trade = None
            
            continue  # Skip new signal if trade open
        
        # Need seq_len bars for prediction
        if len(feature_buffer) < seq_len:
            continue
        
        # Get prediction
        seq = np.array(feature_buffer[-seq_len:])
        seq_norm = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
        
        with torch.no_grad():
            x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Increment cooldown counter
        bars_since_last_trade += 1
        
        # Trade signal (with cooldown check)
        if confidence >= min_confidence and bars_since_last_trade >= min_bars_between_trades:
            direction = 'long' if pred_class == 1 else 'short'
            regime = 'trend_up' if pred_class == 1 else 'trend_down'
            
            # Apply spread to entry price
            entry_price = price + spread if direction == 'long' else price - spread
            
            current_trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                direction=direction,
                entry_price=entry_price,
                exit_price=None,
                pnl=None,
                regime=regime,
                confidence=confidence
            )
            bars_since_last_trade = 0  # Reset cooldown
        
        # Update drawdown
        if balance > max_balance:
            max_balance = balance
        dd = (max_balance - balance) / max_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd
        
        # Progress
        if i % 500 == 0:
            progress = (i - warmup) / (total_bars - warmup) * 100
            print(f"Progress: {progress:.1f}% | Balance: ${balance:,.2f} | Trades: {len(trades)} | DD: {max_drawdown:.2f}%")
    
    # Close any open trade at end
    if current_trade:
        current_trade.exit_price = data.iloc[-1]['close']
        current_trade.pnl = (current_trade.exit_price - current_trade.entry_price) * 100 if current_trade.direction == 'long' else (current_trade.entry_price - current_trade.exit_price) * 100
        current_trade.exit_time = data.index[-1]
        trades.append(current_trade)
        balance += current_trade.pnl
    
    # Results
    print("\n" + "="*60)
    print("BACKTEST RESULTS - Model v3")
    print("="*60)
    
    total_pnl = sum(t.pnl for t in trades if t.pnl)
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses)) if losses else float('inf')
    
    print(f"\n💰 Performance:")
    print(f"   Initial Balance:  ${initial_balance:,.2f}")
    print(f"   Final Balance:    ${balance:,.2f}")
    print(f"   Total P&L:        ${total_pnl:,.2f}")
    print(f"   ROI:              {(total_pnl/initial_balance)*100:.2f}%")
    print(f"   Max Drawdown:     {max_drawdown:.2f}%")
    
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades:     {len(trades)}")
    print(f"   Winning Trades:   {len(wins)}")
    print(f"   Losing Trades:    {len(losses)}")
    print(f"   Win Rate:         {win_rate:.2f}%")
    print(f"   Avg Win:          ${avg_win:,.2f}")
    print(f"   Avg Loss:         ${avg_loss:,.2f}")
    print(f"   Profit Factor:    {profit_factor:.2f}")
    
    # Print trades
    if trades:
        print(f"\n📋 Trade Details (first 20):")
        print("-"*100)
        print(f"{'#':>3} | {'Entry Time':>20} | {'Direction':>6} | {'Entry':>10} | {'Exit':>10} | {'P&L':>10} | {'Conf':>5}")
        print("-"*100)
        for i, t in enumerate(trades[:20], 1):
            entry_time = t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
            pnl_str = f"${t.pnl:+.2f}" if t.pnl else "$0.00"
            print(f"{i:>3} | {entry_time:>20} | {t.direction:>6} | {t.entry_price:>10.2f} | {t.exit_price:>10.2f} | {pnl_str:>10} | {t.confidence:.2f}")
        if len(trades) > 20:
            print(f"... and {len(trades) - 20} more trades")
        print("-"*100)
    
    # Export results
    print("\n[6] Exporting results...")
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'pnl': t.pnl,
        'confidence': t.confidence,
        'regime': t.regime
    } for t in trades])
    trades_df.to_csv('backtest_v3_simple_results.csv', index=False)
    print(f"✅ Results saved to backtest_v3_simple_results.csv")
    
    connector.shutdown()
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()

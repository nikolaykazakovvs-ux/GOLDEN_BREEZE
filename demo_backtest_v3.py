"""
Быстрый бэктест для модели v3 С РЕАЛЬНЫМ INFERENCE (не mock)
"""

from strategy import StrategyConfig, HybridStrategy
from strategy.backtest_engine import BacktestEngine
from mcp_servers.trading import market_data
from mcp_servers.trading.mt5_connector import MT5Connector
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# === LSTM Model Definition ===
class DirectionLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def load_model_v3():
    """Load trained LSTM model v3"""
    model_path = Path("models/direction_lstm_gold_v3.pt")
    metadata_path = Path("models/direction_lstm_gold_v3.json")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None
    
    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Create model (use n_features from v3 metadata)
    n_features = metadata.get('n_features', metadata.get('num_features', 32))
    model = DirectionLSTM(
        input_size=n_features,
        hidden_size=metadata.get('hidden_size', 64),
        num_layers=metadata.get('num_layers', 2),
        num_classes=metadata.get('n_classes', 2),
        dropout=metadata.get('dropout', 0.3)
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    print(f"✅ Model v3 loaded: {n_features} features")
    return model, metadata


def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление индикаторов к данным"""
    data = df.copy()
    
    # SMA Fast (20)
    data["sma_fast"] = data["close"].rolling(window=20).mean()
    
    # SMA Slow (50)
    data["sma_slow"] = data["close"].rolling(window=50).mean()
    
    # ATR (14)
    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(window=14).mean()
    
    # RSI (14)
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))
    
    # Заполнение NaN
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data


def main():
    """
    Backtest Model v3 на данных из MT5 (БЕЗ AI-сервера)
    """
    print("\n" + "="*60)
    print("Golden Breeze - Model v3 Backtest (NO AI SERVER)")
    print("="*60)
    
    # Подключение к MT5
    print("\n[1] Connecting to MT5...")
    connector = MT5Connector()
    if not connector.initialize():
        print("❌ Failed to connect to MT5")
        return
    
    print("✅ Connected to MT5")
    account_info = connector.get_account_info()
    print(f"   Account: {account_info.get('login', 'N/A')}")
    print(f"   Server: {account_info.get('server', 'N/A')}")
    
    # Получение M5 данных (из кеша или live)
    print("\n[2] Loading M5 data...")
    m5_data = market_data.get_ohlcv("XAUUSD", "M5", count=5000)
    
    if m5_data is None or len(m5_data) < 100:
        print("❌ Failed to load M5 data")
        connector.shutdown()
        return
    
    print(f"✅ Loaded {len(m5_data)} M5 bars")
    print(f"   Period: {m5_data.index[0]} to {m5_data.index[-1]}")
    
    # Добавление индикаторов
    print("\n[3] Calculating indicators...")
    m5_data = prepare_data_with_indicators(m5_data)
    print("✅ Indicators added: SMA, ATR, RSI")
    
    # Конфигурация стратегии
    print("\n[4] Configuring strategy (NO AI MODE)...")
    config = StrategyConfig(
        symbol="XAUUSD",
        base_timeframe="M5",
        primary_tf="M5",
        
        # *** CRITICAL: Disable TimeframeSelector (requires M15 data) ***
        tf_selector_enable=False,
        
        risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        max_total_dd_pct=10.0,
        max_positions=3,
        
        # *** Lower confidence threshold to catch more trades ***
        min_direction_confidence=0.60,
        
        ai_api_url="http://127.0.0.1:5005",  # Не используется в mock режиме
        
        # Trend settings
        trend_partial_tp_pct=50.0,
        trend_trailing_atr_mult=2.0,
        trend_min_profit_for_trail=0.5,
        
        # Range settings
        range_tp_fixed_points=100.0,
        range_max_atr_threshold=150.0,
        
        # Volatile settings - allow some trades
        volatile_allow_trades=True,
        volatile_risk_reduction=0.5,
        
        # Backtesting
        use_tick_data=False,
        initial_balance=10000.0
    )
    
    print("✅ Strategy configured")
    print(f"   Initial Balance: ${config.initial_balance:,.2f}")
    print(f"   Risk per Trade: {config.risk_per_trade_pct}%")
    print(f"   Max Daily Loss: {config.max_daily_loss_pct}%")
    
    # Создание стратегии
    print("\n[5] Creating strategy...")
    strategy = HybridStrategy(config, initial_balance=config.initial_balance)
    
    # Load LSTM Model v3
    print("\n[6] Loading Direction LSTM v3...")
    model, metadata = load_model_v3()
    
    if model is None:
        print("❌ Model not loaded, exiting")
        connector.shutdown()
        return
    
    # Create prediction function using real model
    seq_len = 50
    feature_buffer = []
    scaler = StandardScaler()
    scaler_fitted = False
    
    def predict_with_model(features_dict):
        """Real prediction using LSTM v3"""
        nonlocal feature_buffer, scaler, scaler_fitted
        
        # Extract basic features from dict
        close = features_dict.get('close', 0)
        sma_fast = features_dict.get('sma_fast', close)
        sma_slow = features_dict.get('sma_slow', close)
        atr = features_dict.get('atr', 0)
        rsi = features_dict.get('rsi', 50)
        
        # Build simple feature vector (32 features expected)
        feature_vec = [
            close, 
            (close - sma_fast) / max(close, 1e-6) if sma_fast else 0,  # returns proxy
            (close - sma_fast) / max(close, 1e-6) if sma_fast else 0,  # log_returns proxy
            sma_fast if sma_fast else close,
            sma_slow if sma_slow else close,
            sma_fast / max(sma_slow, 1e-6) if sma_slow else 1.0,  # sma_ratio
            atr if atr else 0,
            atr / max(close, 1e-6) if atr else 0,  # atr_norm
            rsi / 100 if rsi else 0.5,
            0.5,  # bb_position placeholder
        ]
        
        # Pad to 32 features
        while len(feature_vec) < 32:
            feature_vec.append(0.0)
        
        feature_buffer.append(feature_vec)
        
        # Keep only last seq_len
        if len(feature_buffer) > seq_len:
            feature_buffer = feature_buffer[-seq_len:]
        
        # Need at least seq_len samples
        if len(feature_buffer) < seq_len:
            return {
                "action": "hold",
                "regime": "range", 
                "direction": "flat",
                "confidence": 0.5
            }
        
        # Prepare sequence
        seq = np.array(feature_buffer[-seq_len:])
        
        # Simple normalization
        seq_normalized = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
        
        # Predict
        with torch.no_grad():
            x = torch.tensor(seq_normalized, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Map prediction to action
        # Class 0 = DOWN, Class 1 = UP (based on training: label-1 mapping)
        if pred_class == 1:  # UP
            direction = "long"
            action = "buy" if confidence >= 0.60 else "hold"
        else:  # DOWN
            direction = "short"
            action = "sell" if confidence >= 0.60 else "hold"
        
        return {
            "action": action,
            "regime": "trend_up" if pred_class == 1 else "trend_down",
            "direction": direction,
            "confidence": confidence,
            "stop_loss": None,
            "take_profit": None
        }
    
    # Override AI client with real model
    def predict_multitf(symbol, timeframes_data):
        """Returns signals for all timeframes"""
        result = {}
        base_signal = predict_with_model({
            'close': m5_data['close'].iloc[-1] if len(m5_data) > 0 else 0,
            'sma_fast': m5_data['sma_fast'].iloc[-1] if 'sma_fast' in m5_data.columns else 0,
            'sma_slow': m5_data['sma_slow'].iloc[-1] if 'sma_slow' in m5_data.columns else 0,
            'atr': m5_data['atr'].iloc[-1] if 'atr' in m5_data.columns else 0,
            'rsi': m5_data['rsi'].iloc[-1] if 'rsi' in m5_data.columns else 50,
        })
        
        # Set signal for all timeframes requested
        for tf in ["M5", "M15", "H1", "H4"]:
            result[tf] = {
                "action": base_signal["action"],
                "regime": base_signal["regime"],
                "direction": base_signal["direction"],
                "direction_confidence": base_signal["confidence"],
                "confidence": base_signal["confidence"],
                "sentiment": "neutral",
                "sentiment_confidence": 0.5,
            }
        return result
    
    strategy.ai_client.predict = predict_with_model
    strategy.ai_client.predict_multitimeframe = predict_multitf
    
    print("✅ AI client connected to LSTM v3 model")
    
    # Backtest engine
    print("\n[7] Initializing backtest engine...")
    backtest = BacktestEngine(strategy, config)
    backtest.load_m5_data(m5_data)
    print("✅ Backtest engine ready")
    
    # Запуск бэктеста
    print("\n[8] Running backtest...")
    print("-"*60)
    try:
        backtest.run()
    except Exception as e:
        print(f"\n⚠️  Backtest error: {e}")
        import traceback
        traceback.print_exc()
    
    # Результаты
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Get trades from risk manager
    trades = strategy.risk_manager.trade_history
    
    # Calculate metrics manually
    total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    final_balance = config.initial_balance + total_pnl
    roi = (total_pnl / config.initial_balance) * 100
    
    print(f"\n💰 Performance:")
    print(f"   Initial Balance: ${config.initial_balance:,.2f}")
    print(f"   Final Balance:   ${final_balance:,.2f}")
    print(f"   Total P&L:       ${total_pnl:,.2f}")
    print(f"   ROI:             {roi:.2f}%")
    
    print(f"\n📊 Trade Statistics:")
    print(f"   Total Trades:    {len(trades)}")
    print(f"   Winning Trades:  {len(wins)}")
    print(f"   Losing Trades:   {len(losses)}")
    print(f"   Win Rate:        {win_rate:.2f}%")
    
    # Print individual trades
    if trades:
        print(f"\n📋 Trade Details:")
        print("-"*80)
        print(f"{'#':>3} | {'Direction':>8} | {'Entry':>10} | {'Exit':>10} | {'P&L':>10} | {'Regime':>12}")
        print("-"*80)
        for i, t in enumerate(trades[:20], 1):  # Show first 20
            pnl = t.pnl if t.pnl else 0
            direction = t.direction if hasattr(t, 'direction') else 'N/A'
            entry = t.entry_price if hasattr(t, 'entry_price') else 0
            exit_p = t.exit_price if hasattr(t, 'exit_price') else 0
            regime = t.regime if hasattr(t, 'regime') else 'N/A'
            pnl_str = f"${pnl:+.2f}" if pnl else "$0.00"
            print(f"{i:>3} | {direction:>8} | {entry:>10.2f} | {exit_p:>10.2f} | {pnl_str:>10} | {regime:>12}")
        if len(trades) > 20:
            print(f"... and {len(trades) - 20} more trades")
        print("-"*80)
    
    # Экспорт результатов
    print("\n[9] Exporting results...")
    backtest.export_results("backtest_v3_results.csv")
    
    # Equity curve
    equity_df = backtest.get_equity_curve()
    equity_df.to_csv("backtest_v3_equity.csv", index=False)
    
    print("✅ Results exported:")
    print("   - backtest_v3_results.csv (trades)")
    print("   - backtest_v3_equity.csv (equity)")
    
    # Отключение MT5
    connector.shutdown()
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()

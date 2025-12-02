"""Example: Train Golden Breeze models using live MT5 data.

Demonstrates end-to-end workflow:
1. Fetch OHLCV data from MT5
2. Train Regime clustering model
3. Train Direction LSTM model
4. Save models for production use
"""
from mcp_servers.trading import market_data
from aimodule.training.train_regime_cluster import train_kmeans
from aimodule.training.train_direction_lstm import train_lstm
from pathlib import Path
import pandas as pd

def main():
    print("=" * 70)
    print("Golden Breeze: Training with Live MT5 Data")
    print("=" * 70)
    
    # Configuration
    SYMBOL = "XAUUSD"
    TIMEFRAME = "M5"
    COUNT = 50000  # Последние 50k свечей (~6 месяцев M5)
    
    # 1. Fetch data from MT5
    print(f"\n[1] Fetching {COUNT} bars of {SYMBOL} {TIMEFRAME} from MT5...")
    df = market_data.get_ohlcv(SYMBOL, TIMEFRAME, count=COUNT)
    
    if df.empty:
        print("❌ Failed to fetch data from MT5")
        return
    
    print(f"✓ Retrieved {len(df)} bars")
    print(f"  Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"  Latest close: {df['close'].iloc[-1]:.2f}")
    
    # Save to CSV for backup
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / f"{SYMBOL.lower()}_{TIMEFRAME.lower()}_mt5.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    
    # 2. Train Regime Model
    print(f"\n[2] Training Regime clustering model (KMeans, 4 clusters)...")
    try:
        train_kmeans(df, n_clusters=4)
        print("✓ Regime model trained and saved")
    except Exception as e:
        print(f"❌ Regime training failed: {e}")
        return
    
    # 3. Train Direction Model
    print(f"\n[3] Training Direction LSTM model...")
    print("  (This may take several minutes on CPU, or seconds on GPU)")
    
    try:
        # Prepare data for LSTM
        from aimodule.data_pipeline.prepare import prepare_direction_data
        
        # Simple train/val split (80/20)
        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()
        
        print(f"  Train: {len(df_train)} bars, Val: {len(df_val)} bars")
        
        # Train LSTM
        train_lstm(
            df_train, 
            df_val,
            epochs=50,
            batch_size=64,
            lr=0.001,
            patience=10
        )
        print("✓ Direction model trained and saved")
    except Exception as e:
        print(f"❌ Direction training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print("\nTrained models:")
    print("  • Regime Model:    models/regime_model.pkl")
    print("  • Direction Model: models/direction_lstm_model.pt")
    print("\nNext steps:")
    print("  1. Start API server: python -m aimodule.server.local_ai_gateway")
    print("  2. Test predictions: python test_ai_core.py")
    print("  3. Run backtests with trained models")
    print("  4. Connect to your trading strategy")

if __name__ == "__main__":
    main()

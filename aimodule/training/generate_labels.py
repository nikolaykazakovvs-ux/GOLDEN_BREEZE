"""
Generate training labels from HybridStrategy backtest results.

Uses existing HybridStrategy + BacktestEngine to run on historical data
and export labels for Direction LSTM training.

Usage:
    python -m aimodule.training.generate_labels \
        --symbol XAUUSD \
        --primary-tf M5 \
        --data-dir data/raw \
        --output data/labels/direction_labels.csv

Author: Golden Breeze Team
Version: 1.1
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Добавляем корневую директорию в PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from strategy import StrategyConfig, HybridStrategy, BacktestEngine
from strategy.risk_manager import Trade


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training labels from HybridStrategy backtest"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="XAUUSD",
        help="Trading symbol"
    )
    parser.add_argument(
        "--primary-tf",
        type=str,
        default="M5",
        help="Primary timeframe for strategy"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory with raw MT5 data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/labels/direction_labels.csv",
        help="Output CSV file for labels"
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Initial backtest balance"
    )
    parser.add_argument(
        "--tf-selector",
        action="store_true",
        help="Enable TimeframeSelector"
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI client (use fallback predictions only)"
    )
    
    return parser.parse_args()


def load_multitf_data(data_dir: Path, symbol: str) -> Dict[str, pd.DataFrame]:
    """
    Load multi-timeframe data from raw directory.
    
    Args:
        data_dir: Base data directory
        symbol: Trading symbol
    
    Returns:
        Dict mapping timeframe to DataFrame
    """
    symbol_dir = data_dir / symbol
    
    if not symbol_dir.exists():
        raise FileNotFoundError(f"Symbol directory not found: {symbol_dir}")
    
    timeframes = ["M1", "M5", "M15", "H1", "H4"]
    multitf_data = {}
    
    for tf in timeframes:
        csv_path = symbol_dir / f"{tf}.csv"
        parquet_path = symbol_dir / f"{tf}.parquet"
        
        # Пробуем загрузить CSV или Parquet
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            multitf_data[tf] = df
            print(f"✅ Loaded {tf}: {len(df)} bars from {csv_path}")
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df.index = pd.to_datetime(df.index)
            multitf_data[tf] = df
            print(f"✅ Loaded {tf}: {len(df)} bars from {parquet_path}")
        else:
            print(f"⚠️  {tf} data not found, skipping")
    
    if not multitf_data:
        raise FileNotFoundError(f"No data files found in {symbol_dir}")
    
    return multitf_data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to DataFrame.
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        DataFrame with indicators
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_fast'] = df['close'].rolling(window=20).mean()
    df['sma_slow'] = df['close'].rolling(window=50).mean()
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    return df


def extract_labels_from_strategy(
    strategy: HybridStrategy,
    backtest: BacktestEngine
) -> pd.DataFrame:
    """
    Extract training labels from strategy trades.
    
    Args:
        strategy: HybridStrategy instance after backtest
        backtest: BacktestEngine instance after backtest
    
    Returns:
        DataFrame with labels
    """
    trades = strategy.risk_manager.trade_history
    
    if not trades:
        print("⚠️  No trades found in backtest")
        return pd.DataFrame()
    
    labels = []
    
    for trade in trades:
        # Direction label: 0=FLAT, 1=LONG, 2=SHORT
        if trade.direction == "long":
            direction_label = 1
        elif trade.direction == "short":
            direction_label = 2
        else:
            direction_label = 0
        
        labels.append({
            'timestamp': trade.entry_time,
            'symbol': trade.symbol,
            'regime': trade.regime,
            'direction_label': direction_label,
            'direction': trade.direction,
            'pnl': trade.pnl if trade.pnl is not None else 0.0,
            'trade_id': trade.id,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price if trade.exit_price else trade.entry_price,
            'volume': trade.volume,
            'reason': trade.reason
        })
    
    df_labels = pd.DataFrame(labels)
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])
    df_labels = df_labels.sort_values('timestamp').reset_index(drop=True)
    
    return df_labels


def run_backtest_for_labels(
    config: StrategyConfig,
    multitf_data: Dict[str, pd.DataFrame],
    initial_balance: float,
    no_ai: bool = False
) -> tuple[HybridStrategy, BacktestEngine]:
    """
    Run backtest to generate labels.
    
    Args:
        config: Strategy configuration
        multitf_data: Multi-timeframe data
        initial_balance: Initial balance
        no_ai: Disable AI client (use fallback predictions only)
    
    Returns:
        (strategy, backtest_engine)
    """
    print("\n" + "="*60)
    print("Running HybridStrategy backtest for label generation...")
    if no_ai:
        print("⚠️  AI client DISABLED - using fallback predictions only")
    print("="*60)
    
    # Добавляем индикаторы ко всем таймфреймам
    for tf in multitf_data:
        print(f"Adding indicators to {tf}...")
        multitf_data[tf] = add_indicators(multitf_data[tf])
    
    # Создаём стратегию
    strategy = HybridStrategy(config, initial_balance=initial_balance)
    
    # Отключаем AI client если нужно
    if no_ai:
        from unittest.mock import Mock
        # Возвращаем fallback predictions без HTTP запросов
        strategy.ai_client.predict = Mock(return_value={
            "action": "hold",
            "regime": "range",
            "direction": "flat",
            "confidence": 0.5,
            "stop_loss": None,
            "take_profit": None
        })
        strategy.ai_client.predict_multitimeframe = Mock(return_value={
            "action": "hold",
            "regime": "range",
            "direction": "flat",
            "confidence": 0.5,
            "H4_trend": "neutral",
            "H1_trend": "neutral",
            "M15_momentum": "weak",
            "M5_entry": "wait"
        })
        print("✅ AI client mocked successfully")
    
    # Создаём backtest engine
    backtest = BacktestEngine(strategy, config)
    
    # Загружаем данные
    backtest.load_multitf_data(multitf_data)
    
    # Загружаем M1 для интрабара (если есть) - только для полного backtest
    # Для генерации меток M1 не нужен, это сильно ускоряет процесс
    if no_ai:
        print("⚠️  Skipping M1 intrabar data for faster label generation")
    elif "M1" in multitf_data:
        backtest.load_m1_data(multitf_data["M1"])
    
    # Запускаем backtest
    try:
        backtest.run()
    except Exception as e:
        print(f"⚠️  Backtest completed with errors: {e}")
        import traceback
        traceback.print_exc()
    
    # Выводим статистику
    backtest._print_results()
    
    return strategy, backtest


def main():
    """Main label generation function."""
    args = parse_args()
    
    print("="*60)
    print("Golden Breeze - Label Generation for Direction LSTM")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Primary TF: {args.primary_tf}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Загружаем данные
    data_dir = Path(args.data_dir)
    
    try:
        multitf_data = load_multitf_data(data_dir, args.symbol)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(f"\nRun export first:")
        print(f"  python -m tools.export_mt5_history --symbol {args.symbol} --start 2024-01-01 --end 2024-06-01")
        sys.exit(1)
    
    # Создаём конфигурацию стратегии
    config = StrategyConfig(
        symbol=args.symbol,
        primary_tf=args.primary_tf,
        tf_selector_enable=args.tf_selector,
        initial_balance=args.initial_balance,
        # Параметры для label generation (можно настроить)
        risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        min_direction_confidence=0.65,
    )
    
    # Запускаем backtest для генерации labels
    strategy, backtest = run_backtest_for_labels(
        config=config,
        multitf_data=multitf_data,
        initial_balance=args.initial_balance,
        no_ai=args.no_ai
    )
    
    # Извлекаем labels
    print("\n" + "="*60)
    print("Extracting labels from trades...")
    print("="*60)
    
    df_labels = extract_labels_from_strategy(strategy, backtest)
    
    if df_labels.empty:
        print("❌ No labels generated (no trades)")
        sys.exit(1)
    
    print(f"✅ Generated {len(df_labels)} labels")
    print(f"\nLabel distribution:")
    print(df_labels['direction_label'].value_counts().sort_index())
    print(f"\nRegime distribution:")
    print(df_labels['regime'].value_counts())
    
    # Сохраняем labels
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_labels.to_csv(output_path, index=False)
    print(f"\n✅ Labels saved to {output_path}")
    
    # Выводим sample
    print(f"\nSample labels:")
    print(df_labels.head(10).to_string())
    
    print("\n" + "="*60)
    print("Label generation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

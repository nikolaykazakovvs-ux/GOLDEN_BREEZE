#!/usr/bin/env python3
"""
Golden Breeze v4 - Backtest Script

Runs backtest on the trained v4 Fusion Transformer model.
Uses V4InferenceAdapter for predictions and simple trade logic.

Usage:
    python -m tools.run_v4_backtest --m5-path data/raw/XAUUSD/M5.csv --h1-path data/raw/XAUUSD/H1.csv

Author: Golden Breeze Team
Version: 4.0.0
Date: 2025-12-04
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aimodule.inference.v4_adapter import V4InferenceAdapter, load_v4_adapter


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    direction: str  # 'BUY' or 'SELL'
    score: float
    confidence: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Data settings
    lookback_bars: int = 5000  # Number of M5 bars to test
    warmup_bars: int = 250     # Bars for model warmup
    
    # Trading settings
    score_threshold: float = 0.3      # Min |score| to enter
    confidence_threshold: float = 0.4  # Min confidence to enter
    
    # Risk management
    take_profit_pips: float = 100.0   # TP in pips
    stop_loss_pips: float = 50.0      # SL in pips
    max_holding_bars: int = 48        # Max bars to hold (4 hours for M5)
    
    # Position sizing
    lot_size: float = 0.1
    pip_value: float = 10.0  # USD per pip for 0.1 lot on XAUUSD
    
    # Initial capital
    initial_balance: float = 10000.0


class V4Backtester:
    """
    Backtester for v4 Fusion Transformer model.
    
    Features:
    - Simple score-based entry logic
    - Fixed TP/SL or max holding time exit
    - Performance metrics calculation
    - Trade log export
    """
    
    def __init__(
        self,
        adapter: V4InferenceAdapter,
        config: BacktestConfig = None,
    ):
        self.adapter = adapter
        self.config = config or BacktestConfig()
        
        # State
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.balance = self.config.initial_balance
        self.equity_curve: List[Dict] = []
        
        # Stats
        self.total_predictions = 0
        self.predictions_by_class = {0: 0, 1: 0, 2: 0}
    
    def run(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict:
        """
        Run backtest on M5/H1 data.
        
        Args:
            df_m5: M5 OHLCV DataFrame
            df_h1: H1 OHLCV DataFrame
            start_idx: Starting bar index (default: -lookback_bars)
            end_idx: Ending bar index (default: end of data)
            
        Returns:
            Dictionary with backtest results
        """
        print("\n" + "=" * 60)
        print("Golden Breeze v4 - Backtest")
        print("=" * 60)
        
        # Prepare data
        if start_idx is None:
            start_idx = len(df_m5) - self.config.lookback_bars
        if end_idx is None:
            end_idx = len(df_m5)
        
        start_idx = max(self.config.warmup_bars, start_idx)
        
        print(f"Testing bars: {start_idx} to {end_idx} ({end_idx - start_idx} bars)")
        print(f"Config:")
        print(f"  Score threshold: {self.config.score_threshold}")
        print(f"  Confidence threshold: {self.config.confidence_threshold}")
        print(f"  TP: {self.config.take_profit_pips} pips")
        print(f"  SL: {self.config.stop_loss_pips} pips")
        print(f"  Max hold: {self.config.max_holding_bars} bars")
        print("=" * 60)
        
        # Ensure time column
        if 'time' not in df_m5.columns:
            df_m5 = df_m5.reset_index()
            df_m5.columns = ['time'] + list(df_m5.columns[1:])
        if 'time' not in df_h1.columns:
            df_h1 = df_h1.reset_index()
            df_h1.columns = ['time'] + list(df_h1.columns[1:])
        
        df_m5['time'] = pd.to_datetime(df_m5['time'])
        df_h1['time'] = pd.to_datetime(df_h1['time'])
        
        # Main backtest loop
        total_bars = end_idx - start_idx
        progress_step = max(1, total_bars // 20)
        
        for i in range(start_idx, end_idx):
            # Current bar
            current_bar = df_m5.iloc[i]
            current_time = current_bar['time']
            current_close = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            
            # Get historical context
            m5_history = df_m5.iloc[:i + 1]
            h1_mask = df_h1['time'] <= current_time
            h1_history = df_h1[h1_mask]
            
            # Check minimum data requirements
            if len(m5_history) < self.adapter.config.seq_len_fast:
                continue
            if len(h1_history) < self.adapter.config.seq_len_slow:
                continue
            
            # Check existing position
            if self.current_trade:
                self._check_exit(current_bar, i - start_idx)
            
            # Generate prediction (only if no position)
            if not self.current_trade:
                try:
                    prediction = self.adapter.predict(m5_history, h1_history)
                    self.total_predictions += 1
                    self.predictions_by_class[prediction['class']] += 1
                    
                    # Entry logic
                    self._check_entry(current_bar, prediction)
                    
                except Exception as e:
                    # Skip on error
                    if i % 1000 == 0:
                        print(f"⚠️  Prediction error at bar {i}: {e}")
            
            # Record equity
            unrealized_pnl = 0
            if self.current_trade:
                if self.current_trade.direction == 'BUY':
                    unrealized_pnl = (current_close - self.current_trade.entry_price) * self.config.pip_value
                else:
                    unrealized_pnl = (self.current_trade.entry_price - current_close) * self.config.pip_value
            
            self.equity_curve.append({
                'time': current_time,
                'balance': self.balance,
                'equity': self.balance + unrealized_pnl,
                'position': 1 if self.current_trade else 0,
            })
            
            # Progress
            if (i - start_idx) % progress_step == 0:
                pct = (i - start_idx) / total_bars * 100
                print(f"Progress: {pct:.0f}% | Balance: ${self.balance:,.2f} | "
                      f"Trades: {len(self.trades)}")
        
        # Close any open trade at end
        if self.current_trade:
            last_bar = df_m5.iloc[end_idx - 1]
            self._close_trade(last_bar, "END_OF_DATA")
        
        # Calculate results
        results = self._calculate_results()
        self._print_results(results)
        
        return results
    
    def _check_entry(self, bar: pd.Series, prediction: Dict):
        """Check entry conditions and open trade if met."""
        score = prediction['score']
        confidence = prediction['confidence']
        
        # Check thresholds
        if abs(score) < self.config.score_threshold:
            return
        if confidence < self.config.confidence_threshold:
            return
        
        # Determine direction
        if score > 0:
            direction = 'BUY'
            entry_price = bar['close'] + 0.3  # Simulate spread
        else:
            direction = 'SELL'
            entry_price = bar['close'] - 0.3  # Simulate spread
        
        # Open trade
        self.current_trade = Trade(
            entry_time=bar['time'],
            entry_price=entry_price,
            direction=direction,
            score=score,
            confidence=confidence,
        )
    
    def _check_exit(self, bar: pd.Series, bars_held: int):
        """Check exit conditions for current trade."""
        if not self.current_trade:
            return
        
        current_close = bar['close']
        trade = self.current_trade
        
        # Calculate current P/L in pips
        if trade.direction == 'BUY':
            pnl_pips = current_close - trade.entry_price
            hit_tp = bar['high'] >= trade.entry_price + self.config.take_profit_pips
            hit_sl = bar['low'] <= trade.entry_price - self.config.stop_loss_pips
        else:
            pnl_pips = trade.entry_price - current_close
            hit_tp = bar['low'] <= trade.entry_price - self.config.take_profit_pips
            hit_sl = bar['high'] >= trade.entry_price + self.config.stop_loss_pips
        
        # Check exit conditions
        if hit_tp:
            exit_price = trade.entry_price + (self.config.take_profit_pips if trade.direction == 'BUY' else -self.config.take_profit_pips)
            self._close_trade(bar, "TAKE_PROFIT", exit_price)
        elif hit_sl:
            exit_price = trade.entry_price + (-self.config.stop_loss_pips if trade.direction == 'BUY' else self.config.stop_loss_pips)
            self._close_trade(bar, "STOP_LOSS", exit_price)
        elif bars_held >= self.config.max_holding_bars:
            self._close_trade(bar, "MAX_TIME")
    
    def _close_trade(self, bar: pd.Series, reason: str, exit_price: float = None):
        """Close current trade and record results."""
        if not self.current_trade:
            return
        
        trade = self.current_trade
        trade.exit_time = bar['time']
        trade.exit_price = exit_price if exit_price else bar['close']
        trade.exit_reason = reason
        
        # Calculate P/L
        if trade.direction == 'BUY':
            pnl_pips = trade.exit_price - trade.entry_price
        else:
            pnl_pips = trade.entry_price - trade.exit_price
        
        trade.pnl = pnl_pips * self.config.pip_value
        
        # Update balance
        self.balance += trade.pnl
        
        # Record trade
        self.trades.append(trade)
        self.current_trade = None
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'final_balance': self.balance,
                'net_pnl': 0,
                'roi_pct': 0,
            }
        
        # Trade stats
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        # Averages
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown
        equity = pd.DataFrame(self.equity_curve)['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        # By exit reason
        reason_stats = {}
        for reason in ['TAKE_PROFIT', 'STOP_LOSS', 'MAX_TIME', 'END_OF_DATA']:
            reason_trades = [t for t in self.trades if t.exit_reason == reason]
            reason_stats[reason] = {
                'count': len(reason_trades),
                'pnl': sum(t.pnl for t in reason_trades),
            }
        
        # Prediction distribution
        total_pred = sum(self.predictions_by_class.values())
        pred_dist = {
            k: v / total_pred * 100 if total_pred > 0 else 0 
            for k, v in self.predictions_by_class.items()
        }
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'final_balance': self.balance,
            'net_pnl': total_pnl,
            'roi_pct': (total_pnl / self.config.initial_balance) * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd,
            'exit_reasons': reason_stats,
            'prediction_distribution': pred_dist,
            'total_predictions': self.total_predictions,
        }
    
    def _print_results(self, results: Dict):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\n📊 Performance:")
        print(f"  Initial Balance:  ${self.config.initial_balance:,.2f}")
        print(f"  Final Balance:    ${results['final_balance']:,.2f}")
        print(f"  Net P/L:          ${results['net_pnl']:,.2f}")
        print(f"  ROI:              {results['roi_pct']:.2f}%")
        print(f"  Max Drawdown:     {results['max_drawdown_pct']:.2f}%")
        
        print(f"\n📈 Trade Statistics:")
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Wins:             {results['wins']}")
        print(f"  Losses:           {results['losses']}")
        print(f"  Win Rate:         {results['win_rate']:.1f}%")
        print(f"  Avg Win:          ${results['avg_win']:.2f}")
        print(f"  Avg Loss:         ${results['avg_loss']:.2f}")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}")
        
        print(f"\n🎯 Exit Reasons:")
        for reason, stats in results['exit_reasons'].items():
            if stats['count'] > 0:
                print(f"  {reason:15s}: {stats['count']:3d} trades, ${stats['pnl']:8,.2f}")
        
        print(f"\n🔮 Predictions:")
        print(f"  Total: {results['total_predictions']}")
        print(f"  Distribution:")
        for cls, pct in results['prediction_distribution'].items():
            label = {0: 'DOWN', 1: 'HOLD', 2: 'UP'}[cls]
            print(f"    {label}: {pct:.1f}%")
        
        print("\n" + "=" * 60)
    
    def export_trades(self, filepath: str = "reports/v4_backtest_trades.csv"):
        """Export trade log to CSV."""
        if not self.trades:
            print("No trades to export")
            return
        
        df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'score': t.score,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason,
            }
            for t in self.trades
        ])
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Trades exported to {filepath}")
    
    def export_equity(self, filepath: str = "reports/v4_backtest_equity.csv"):
        """Export equity curve to CSV."""
        if not self.equity_curve:
            print("No equity data to export")
            return
        
        df = pd.DataFrame(self.equity_curve)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✅ Equity exported to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run v4 Fusion Transformer backtest")
    parser.add_argument('--m5-path', type=str, default='data/raw/XAUUSD/M5.csv',
                        help='Path to M5 CSV data')
    parser.add_argument('--h1-path', type=str, default='data/raw/XAUUSD/H1.csv',
                        help='Path to H1 CSV data')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model checkpoint (auto-detect if not provided)')
    parser.add_argument('--bars', type=int, default=5000,
                        help='Number of bars to test')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Minimum score threshold')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                        help='Minimum confidence threshold')
    parser.add_argument('--tp', type=float, default=100.0,
                        help='Take profit in pips')
    parser.add_argument('--sl', type=float, default=50.0,
                        help='Stop loss in pips')
    parser.add_argument('--export', action='store_true',
                        help='Export trades and equity to CSV')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Golden Breeze v4 - Backtest Runner")
    print("=" * 60)
    
    # Load adapter
    print("\n📦 Loading model...")
    try:
        adapter = load_v4_adapter(args.model_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please train the model first or specify --model-path")
        return 1
    
    # Load data
    print("\n📂 Loading data...")
    df_m5 = pd.read_csv(args.m5_path)
    df_h1 = pd.read_csv(args.h1_path)
    print(f"  M5: {len(df_m5)} bars")
    print(f"  H1: {len(df_h1)} bars")
    
    # Configure backtest
    config = BacktestConfig(
        lookback_bars=args.bars,
        score_threshold=args.score_threshold,
        confidence_threshold=args.confidence_threshold,
        take_profit_pips=args.tp,
        stop_loss_pips=args.sl,
    )
    
    # Run backtest
    backtester = V4Backtester(adapter, config)
    results = backtester.run(df_m5, df_h1)
    
    # Export if requested
    if args.export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backtester.export_trades(f"reports/v4_backtest_trades_{timestamp}.csv")
        backtester.export_equity(f"reports/v4_backtest_equity_{timestamp}.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

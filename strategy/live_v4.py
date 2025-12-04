"""
Golden Breeze V4 - Live Trading Engine

Real-time trading engine using LSTM V4 model for XAUUSD.

Signal Logic:
    - UP (Class 2) + prob > 0.55: Open BUY
    - DOWN (Class 0) + prob > 0.55: Open SELL  
    - NEUTRAL (Class 1) + prob > 0.60: Close existing positions

Usage:
    python strategy/live_v4.py --paper     # Paper trading (no real execution)
    python strategy/live_v4.py             # Live trading

Author: Golden Breeze Team
Version: 4.1.0
Date: 2025-12-04
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available. Paper trading only.")

from aimodule.inference.lstm_v4_adapter import LSTMV4Adapter, PredictionResult


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradingConfig:
    """Trading configuration."""
    
    # Symbol
    symbol: str = "XAUUSD"
    
    # Model
    model_path: str = "models/v4_5class/lstm_3class_best.pt"
    
    # Data requirements
    m5_bars_required: int = 200
    h1_bars_required: int = 50
    
    # Signal thresholds
    buy_threshold: float = 0.55       # Probability threshold for BUY
    sell_threshold: float = 0.55      # Probability threshold for SELL
    neutral_threshold: float = 0.60   # Probability threshold for NEUTRAL (close)
    
    # Risk management
    fixed_lot: float = 0.01           # Fixed lot size for testing
    max_positions: int = 1            # Maximum concurrent positions
    sl_pips: float = 50.0             # Stop loss in pips (for XAUUSD: $0.50)
    tp_pips: float = 100.0            # Take profit in pips
    
    # Timing
    check_interval: float = 1.0       # Seconds between checks
    
    # Trading hours (UTC)
    session_start_hour: int = 1       # 01:00 UTC
    session_end_hour: int = 22        # 22:00 UTC
    
    # Logging
    log_file: str = "logs/live_v4.log"


# =============================================================================
# MT5 CONNECTOR
# =============================================================================

class MT5Connector:
    """MetaTrader 5 connector."""
    
    def __init__(self, config: TradingConfig, paper_mode: bool = True):
        self.config = config
        self.paper_mode = paper_mode
        self.connected = False
        
        # Paper trading state
        self.paper_positions: Dict[int, Dict] = {}
        self.paper_ticket_counter = 1000
        
    def connect(self) -> bool:
        """Connect to MT5."""
        if self.paper_mode:
            self.connected = True
            return True
        
        if not MT5_AVAILABLE:
            print("❌ MT5 not available")
            return False
        
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from MT5."""
        if not self.paper_mode and MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
    
    def get_m5_data(self, bars: int = 200) -> Optional[pd.DataFrame]:
        """Get M5 OHLCV data."""
        if self.paper_mode:
            # Load from file for paper trading
            path = f"data/raw/{self.config.symbol}/M5.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df.tail(bars)
            return None
        
        if not MT5_AVAILABLE:
            return None
        
        rates = mt5.copy_rates_from_pos(
            self.config.symbol, 
            mt5.TIMEFRAME_M5, 
            0, 
            bars
        )
        
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_h1_data(self, bars: int = 50) -> Optional[pd.DataFrame]:
        """Get H1 OHLCV data."""
        if self.paper_mode:
            path = f"data/raw/{self.config.symbol}/H1.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df.tail(bars)
            return None
        
        if not MT5_AVAILABLE:
            return None
        
        rates = mt5.copy_rates_from_pos(
            self.config.symbol, 
            mt5.TIMEFRAME_H1, 
            0, 
            bars
        )
        
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_current_price(self) -> Tuple[float, float]:
        """Get current bid/ask."""
        if self.paper_mode:
            df = self.get_m5_data(1)
            if df is not None and len(df) > 0:
                close = df['close'].iloc[-1]
                spread = 0.30  # Typical XAUUSD spread
                return close, close + spread
            return 0.0, 0.0
        
        if not MT5_AVAILABLE:
            return 0.0, 0.0
        
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return 0.0, 0.0
        
        return tick.bid, tick.ask
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if self.paper_mode:
            return list(self.paper_positions.values())
        
        if not MT5_AVAILABLE:
            return []
        
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None:
            return []
        
        return [
            {
                'ticket': p.ticket,
                'type': 'BUY' if p.type == 0 else 'SELL',
                'volume': p.volume,
                'price_open': p.price_open,
                'sl': p.sl,
                'tp': p.tp,
                'profit': p.profit,
                'time': datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]
    
    def open_position(
        self, 
        direction: str,  # 'BUY' or 'SELL'
        volume: float,
        sl: float,
        tp: float,
        comment: str = "LSTM_V4",
    ) -> Optional[int]:
        """Open a position. Returns ticket or None."""
        bid, ask = self.get_current_price()
        
        if direction == 'BUY':
            price = ask
        else:
            price = bid
        
        if self.paper_mode:
            ticket = self.paper_ticket_counter
            self.paper_ticket_counter += 1
            
            self.paper_positions[ticket] = {
                'ticket': ticket,
                'type': direction,
                'volume': volume,
                'price_open': price,
                'sl': sl,
                'tp': tp,
                'profit': 0.0,
                'time': datetime.now(),
                'comment': comment,
            }
            
            return ticket
        
        if not MT5_AVAILABLE:
            return None
        
        # Prepare request
        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result.comment}")
            return None
        
        return result.order
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket."""
        if self.paper_mode:
            if ticket in self.paper_positions:
                del self.paper_positions[ticket]
                return True
            return False
        
        if not MT5_AVAILABLE:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        
        # Prepare close request
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.config.symbol).bid if position.type == 0 else mt5.symbol_info_tick(self.config.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "LSTM_V4_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def close_all_positions(self) -> int:
        """Close all positions. Returns count of closed."""
        positions = self.get_positions()
        closed = 0
        
        for pos in positions:
            if self.close_position(pos['ticket']):
                closed += 1
        
        return closed


# =============================================================================
# LIVE TRADING ENGINE
# =============================================================================

class LiveTradingEngineV4:
    """
    Live trading engine using LSTM V4 model.
    
    Flow:
    1. Check if new M5 bar closed
    2. Fetch M5 and H1 data
    3. Run prediction
    4. Execute trade based on signal
    """
    
    def __init__(
        self,
        config: TradingConfig,
        paper_mode: bool = True,
    ):
        self.config = config
        self.paper_mode = paper_mode
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.logger.info("=" * 60)
        self.logger.info("🚀 Golden Breeze V4 - Live Trading Engine")
        self.logger.info("=" * 60)
        self.logger.info(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
        self.logger.info(f"Symbol: {config.symbol}")
        self.logger.info(f"Model: {config.model_path}")
        
        # Load model
        self.logger.info("Loading LSTM V4 model...")
        self.adapter = LSTMV4Adapter(config.model_path)
        
        # Initialize MT5 connector
        self.mt5 = MT5Connector(config, paper_mode)
        
        # State
        self.last_bar_time: Optional[datetime] = None
        self.running = False
        self.signals_count = 0
        self.trades_count = 0
        
    def _setup_logging(self):
        """Setup logging."""
        os.makedirs(os.path.dirname(self.config.log_file), exist_ok=True)
        
        self.logger = logging.getLogger("LiveV4")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(self.config.log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _is_trading_session(self) -> bool:
        """Check if within trading hours."""
        now = datetime.now(tz=None)  # Local time
        return self.config.session_start_hour <= now.hour < self.config.session_end_hour
    
    def _detect_new_bar(self, df_m5: pd.DataFrame) -> bool:
        """Detect if a new M5 bar has closed."""
        if df_m5 is None or len(df_m5) == 0:
            return False
        
        # Get last bar time
        if 'time' in df_m5.columns:
            last_time = pd.to_datetime(df_m5['time'].iloc[-1])
        else:
            return False
        
        if self.last_bar_time is None:
            self.last_bar_time = last_time
            return True  # First bar
        
        if last_time > self.last_bar_time:
            self.last_bar_time = last_time
            return True
        
        return False
    
    def _calculate_sl_tp(
        self, 
        direction: str, 
        price: float,
    ) -> Tuple[float, float]:
        """Calculate SL and TP levels."""
        if direction == 'BUY':
            sl = price - self.config.sl_pips
            tp = price + self.config.tp_pips
        else:  # SELL
            sl = price + self.config.sl_pips
            tp = price - self.config.tp_pips
        
        return round(sl, 2), round(tp, 2)
    
    def _process_signal(self, prediction: PredictionResult):
        """Process prediction signal and execute trade if needed."""
        positions = self.mt5.get_positions()
        has_position = len(positions) > 0
        
        pred_class = prediction.pred_class
        prob = prediction.confidence
        label = prediction.label
        
        # Log signal
        self.signals_count += 1
        probs_str = f"D:{prediction.probs[0]:.1%} N:{prediction.probs[1]:.1%} U:{prediction.probs[2]:.1%}"
        self.logger.info(f"📊 Signal #{self.signals_count}: {label} ({prob:.1%}) [{probs_str}]")
        
        # Decision logic
        action = None
        
        if pred_class == 2 and prob >= self.config.buy_threshold:
            # UP signal
            if not has_position:
                action = 'BUY'
            elif positions[0]['type'] == 'SELL':
                # Close SELL, open BUY
                self.logger.info("🔄 Reversing SELL -> BUY")
                self.mt5.close_position(positions[0]['ticket'])
                action = 'BUY'
        
        elif pred_class == 0 and prob >= self.config.sell_threshold:
            # DOWN signal
            if not has_position:
                action = 'SELL'
            elif positions[0]['type'] == 'BUY':
                # Close BUY, open SELL
                self.logger.info("🔄 Reversing BUY -> SELL")
                self.mt5.close_position(positions[0]['ticket'])
                action = 'SELL'
        
        elif pred_class == 1 and prob >= self.config.neutral_threshold:
            # NEUTRAL signal - close positions
            if has_position:
                self.logger.info("🔴 NEUTRAL signal - closing positions")
                closed = self.mt5.close_all_positions()
                if closed > 0:
                    self.logger.info(f"   Closed {closed} position(s)")
        
        # Execute action
        if action:
            bid, ask = self.mt5.get_current_price()
            price = ask if action == 'BUY' else bid
            sl, tp = self._calculate_sl_tp(action, price)
            
            self.logger.info(f"📈 EXECUTING {action} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            ticket = self.mt5.open_position(
                direction=action,
                volume=self.config.fixed_lot,
                sl=sl,
                tp=tp,
                comment=f"LSTM_V4_{label}",
            )
            
            if ticket:
                self.trades_count += 1
                self.logger.info(f"✅ Order executed: Ticket #{ticket}")
            else:
                self.logger.error("❌ Order execution failed")
    
    def run(self):
        """Main trading loop."""
        # Connect to MT5
        if not self.mt5.connect():
            self.logger.error("Failed to connect to MT5")
            return
        
        self.logger.info("✅ Connected to MT5")
        self.running = True
        
        try:
            while self.running:
                # Check trading session
                if not self._is_trading_session():
                    self.logger.debug("Outside trading session, waiting...")
                    time.sleep(60)
                    continue
                
                # Fetch data
                df_m5 = self.mt5.get_m5_data(self.config.m5_bars_required)
                
                if df_m5 is None:
                    self.logger.warning("Failed to get M5 data")
                    time.sleep(self.config.check_interval)
                    continue
                
                # Check for new bar
                if not self._detect_new_bar(df_m5):
                    # No new bar, just wait
                    time.sleep(self.config.check_interval)
                    continue
                
                self.logger.info("-" * 40)
                self.logger.info(f"🕐 New M5 bar: {self.last_bar_time}")
                
                # Fetch H1 data
                df_h1 = self.mt5.get_h1_data(self.config.h1_bars_required)
                
                if df_h1 is None:
                    self.logger.warning("Failed to get H1 data")
                    continue
                
                # Run prediction
                try:
                    prediction = self.adapter.predict(df_m5, df_h1)
                    self._process_signal(prediction)
                except Exception as e:
                    self.logger.error(f"Prediction error: {e}")
                
                # Wait for next check
                time.sleep(self.config.check_interval)
        
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Interrupted by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the engine."""
        self.running = False
        
        self.logger.info("=" * 60)
        self.logger.info("📊 Session Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Signals processed: {self.signals_count}")
        self.logger.info(f"Trades executed: {self.trades_count}")
        
        # Show open positions
        positions = self.mt5.get_positions()
        if positions:
            self.logger.info(f"Open positions: {len(positions)}")
            for pos in positions:
                self.logger.info(f"  {pos['type']} {pos['volume']} @ {pos['price_open']:.2f}")
        
        self.mt5.disconnect()
        self.logger.info("✅ Engine stopped")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Golden Breeze V4 - Live Trading Engine"
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode (no real execution)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/v4_5class/lstm_3class_best.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--lot',
        type=float,
        default=0.01,
        help='Fixed lot size',
    )
    parser.add_argument(
        '--buy-threshold',
        type=float,
        default=0.55,
        help='Probability threshold for BUY signal',
    )
    parser.add_argument(
        '--sell-threshold',
        type=float,
        default=0.55,
        help='Probability threshold for SELL signal',
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='XAUUSD',
        help='Trading symbol',
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TradingConfig(
        symbol=args.symbol,
        model_path=args.model,
        fixed_lot=args.lot,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )
    
    # Create and run engine
    engine = LiveTradingEngineV4(
        config=config,
        paper_mode=args.paper,
    )
    
    engine.run()


if __name__ == "__main__":
    main()

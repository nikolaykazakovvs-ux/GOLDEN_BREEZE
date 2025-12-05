#!/usr/bin/env python3
"""
🏆 Golden Breeze v5 Ultimate - Live Trading Bot
================================================

This is the production-ready live trading bot powered by:
- GoldenBreezeV5Ultimate model (MCC +0.5729, 327K params)
- Multi-timeframe analysis (M5 + H1)
- Confidence-based signal filtering
- Online Learning (continuous improvement)

Author: Golden Breeze Team
Version: 5.0.1 Ultimate
Date: 2025-12-05

Usage:
    python live_v5_ultimate.py
    
Requirements:
    - MetaTrader5 terminal running
    - Trained model in models/v5_ultimate/
    - XAUUSD symbol available
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import MetaTrader5 as mt5

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from aimodule.inference.v5_adapter import GoldenBreezeAdapter, PredictionResult
from aimodule.learning.online_learning import OnlineLearningBuffer, get_learning_buffer


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Live trading configuration."""
    
    # Symbol settings
    SYMBOL = "XAUUSD"
    TIMEFRAME_FAST = mt5.TIMEFRAME_M5
    TIMEFRAME_SLOW = mt5.TIMEFRAME_H1
    
    # Data settings
    BARS_M5 = 300   # Buffer for indicators (need ~50 for model)
    BARS_H1 = 100   # Buffer for H1 indicators (need ~20 for model)
    
    # Trading thresholds
    CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence for signals
    STRONG_SIGNAL_THRESHOLD = 0.75  # Strong signal level
    
    # Risk management (for future)
    LOT_SIZE = 0.01
    STOP_LOSS_PIPS = 50
    TAKE_PROFIT_PIPS = 100
    
    # Logging
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_FILE = LOG_DIR / "live_v5.csv"
    
    # Timing
    LOOP_INTERVAL_SECONDS = 10  # Check every 10 seconds
    HEARTBEAT_INTERVAL = 60     # Print heartbeat every minute
    
    # Resilience settings
    MAX_RECONNECT_ATTEMPTS = 10
    RECONNECT_DELAY_SECONDS = 5
    MAX_CONSECUTIVE_ERRORS = 5
    ERROR_COOLDOWN_SECONDS = 30


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging."""
    Config.LOG_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Config.LOG_DIR / "live_v5_ultimate.log")
        ]
    )
    return logging.getLogger("GoldenBreezeV5")


logger = setup_logging()


# =============================================================================
# MT5 CONNECTION
# =============================================================================

def is_mt5_connected() -> bool:
    """Check if MT5 is still connected."""
    try:
        info = mt5.terminal_info()
        return info is not None and info.connected
    except:
        return False


def reconnect_mt5(max_attempts: int = None) -> bool:
    """
    Attempt to reconnect to MT5 with exponential backoff.
    
    Returns:
        True if reconnected, False if all attempts failed.
    """
    if max_attempts is None:
        max_attempts = Config.MAX_RECONNECT_ATTEMPTS
    
    for attempt in range(1, max_attempts + 1):
        logger.warning(f"🔄 Reconnect attempt {attempt}/{max_attempts}...")
        
        # Shutdown existing connection
        try:
            mt5.shutdown()
        except:
            pass
        
        time.sleep(Config.RECONNECT_DELAY_SECONDS * attempt)  # Exponential backoff
        
        if connect_mt5():
            logger.info(f"✅ Reconnected on attempt {attempt}")
            return True
        
        logger.error(f"❌ Attempt {attempt} failed")
    
    return False


def connect_mt5() -> bool:
    """Initialize MT5 connection."""
    logger.info("🔌 Connecting to MetaTrader 5...")
    
    if not mt5.initialize():
        logger.error(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("❌ Failed to get account info")
        return False
    
    logger.info(f"✅ Connected to MT5")
    logger.info(f"   Account: {account_info.login}")
    logger.info(f"   Server: {account_info.server}")
    logger.info(f"   Balance: ${account_info.balance:,.2f}")
    logger.info(f"   Leverage: 1:{account_info.leverage}")
    
    # Check symbol
    symbol_info = mt5.symbol_info(Config.SYMBOL)
    if symbol_info is None:
        logger.error(f"❌ Symbol {Config.SYMBOL} not found")
        return False
    
    if not symbol_info.visible:
        if not mt5.symbol_select(Config.SYMBOL, True):
            logger.error(f"❌ Failed to select {Config.SYMBOL}")
            return False
    
    logger.info(f"   Symbol: {Config.SYMBOL} (spread: {symbol_info.spread} points)")
    
    return True


def get_mt5_data(symbol: str, timeframe: int, bars: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    
    if rates is None or len(rates) == 0:
        logger.warning(f"⚠️ No data for {symbol} TF={timeframe}")
        return None
    
    df = pd.DataFrame(rates)
    # MT5 returns UTC timestamps
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={
        'time': 'timestamp',
        'tick_volume': 'volume'
    })
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


# =============================================================================
# SIGNAL LOGGING
# =============================================================================

def init_signal_log():
    """Initialize signal log CSV."""
    if not Config.LOG_FILE.exists():
        header = "timestamp,close_price,signal,confidence,prob_down,prob_neutral,prob_up,action\n"
        Config.LOG_FILE.write_text(header)
        logger.info(f"📝 Created signal log: {Config.LOG_FILE}")


def log_signal(
    timestamp: datetime,
    close_price: float,
    result: PredictionResult,
    action: str = "HOLD"
):
    """Append signal to CSV log."""
    row = (
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')},"
        f"{close_price:.2f},"
        f"{result.signal},"
        f"{result.confidence:.4f},"
        f"{result.probabilities['DOWN']:.4f},"
        f"{result.probabilities['NEUTRAL']:.4f},"
        f"{result.probabilities['UP']:.4f},"
        f"{action}\n"
    )
    
    with open(Config.LOG_FILE, 'a') as f:
        f.write(row)


# =============================================================================
# TRADING LOGIC
# =============================================================================

def evaluate_signal(result: PredictionResult) -> Tuple[str, str]:
    """
    Evaluate prediction and determine action.
    
    Returns:
        (action, emoji) tuple
    """
    signal = result.signal
    conf = result.confidence
    
    # Strong signals
    if signal == "UP" and conf >= Config.STRONG_SIGNAL_THRESHOLD:
        return "STRONG_BUY", "🚀🚀"
    
    if signal == "DOWN" and conf >= Config.STRONG_SIGNAL_THRESHOLD:
        return "STRONG_SELL", "🔻🔻"
    
    # Normal signals
    if signal == "UP" and conf >= Config.CONFIDENCE_THRESHOLD:
        return "BUY", "🟢"
    
    if signal == "DOWN" and conf >= Config.CONFIDENCE_THRESHOLD:
        return "SELL", "🔴"
    
    # Weak or neutral
    if signal == "NEUTRAL":
        return "HOLD", "⏸️"
    
    return "WAIT", "⏳"


def print_prediction(
    timestamp: datetime,
    close_price: float,
    result: PredictionResult,
    action: str,
    emoji: str
):
    """Print formatted prediction."""
    probs = result.probabilities
    
    # Build probability bar
    bar_down = "█" * int(probs['DOWN'] * 20)
    bar_neut = "█" * int(probs['NEUTRAL'] * 20)
    bar_up = "█" * int(probs['UP'] * 20)
    
    print("\n" + "=" * 70)
    print(f"{emoji} {action} | {timestamp.strftime('%H:%M:%S')} | {Config.SYMBOL} @ {close_price:.2f}")
    print("=" * 70)
    print(f"   Signal: {result.signal:<8} | Confidence: {result.confidence:.1%}")
    print("-" * 70)
    print(f"   DOWN:    {probs['DOWN']:>6.1%} |{bar_down}")
    print(f"   NEUTRAL: {probs['NEUTRAL']:>6.1%} |{bar_neut}")
    print(f"   UP:      {probs['UP']:>6.1%} |{bar_up}")
    print("=" * 70)


def execute_trade(action: str, price: float):
    """
    Execute trade on MT5.
    
    ⚠️ CURRENTLY DISABLED FOR SAFETY - Enable when ready!
    """
    logger.warning(f"📋 Trade signal: {action} @ {price:.2f} (execution disabled)")
    
    # TODO: Uncomment when ready for live trading
    # if action in ["BUY", "STRONG_BUY"]:
    #     request = {
    #         "action": mt5.TRADE_ACTION_DEAL,
    #         "symbol": Config.SYMBOL,
    #         "volume": Config.LOT_SIZE,
    #         "type": mt5.ORDER_TYPE_BUY,
    #         "price": mt5.symbol_info_tick(Config.SYMBOL).ask,
    #         "sl": price - Config.STOP_LOSS_PIPS * 0.1,
    #         "tp": price + Config.TAKE_PROFIT_PIPS * 0.1,
    #         "magic": 505050,
    #         "comment": "GBv5Ultimate",
    #     }
    #     result = mt5.order_send(request)
    #     logger.info(f"Order result: {result}")


# =============================================================================
# MAIN LOOP
# =============================================================================

def wait_for_new_bar(last_bar_time: Optional[datetime]) -> Optional[datetime]:
    """
    Wait for a new M5 bar to close.
    
    Returns:
        New bar time, or None if connection lost.
    """
    check_count = 0
    while True:
        try:
            # Check connection every 10 iterations
            if check_count % 10 == 0 and not is_mt5_connected():
                logger.warning("⚠️ MT5 connection lost during wait")
                return None
            
            rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME_FAST, 0, 1)
            if rates is not None and len(rates) > 0:
                # Use UTC time (MT5 returns UTC timestamps)
                current_bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)
                
                if last_bar_time is None or current_bar_time > last_bar_time:
                    return current_bar_time
            
            check_count += 1
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"⚠️ Error in wait_for_new_bar: {e}")
            return None


def run_live_bot():
    """Main bot loop."""
    
    print("\n" + "=" * 70)
    print("🏆 GOLDEN BREEZE v5 ULTIMATE - LIVE TRADING BOT")
    print("=" * 70)
    print(f"   Model: GoldenBreezeV5Ultimate (MCC +0.5729)")
    print(f"   Symbol: {Config.SYMBOL}")
    print(f"   Timeframes: M5 (fast) + H1 (slow)")
    print(f"   Confidence Threshold: {Config.CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 70 + "\n")
    
    # Initialize MT5
    if not connect_mt5():
        logger.error("Failed to connect to MT5. Exiting.")
        return
    
    # Initialize adapter (loads model & scalers)
    logger.info("🧠 Initializing v5 Ultimate brain...")
    try:
        adapter = GoldenBreezeAdapter()
    except Exception as e:
        logger.error(f"❌ Failed to initialize adapter: {e}")
        mt5.shutdown()
        return
    
    # Initialize signal log
    init_signal_log()
    
    # Initialize online learning buffer
    learning_buffer = get_learning_buffer()
    
    print("\n" + "🟢 " * 20)
    print("🟢 SYSTEM READY. v5 ULTIMATE ACTIVE. 🟢")
    print("🟢 Online Learning: ENABLED 🟢")
    print("🟢 " * 20 + "\n")
    
    logger.info("Starting main loop... Press Ctrl+C to stop.")
    
    last_bar_time = None
    last_heartbeat = time.time()
    signals_today = {"BUY": 0, "SELL": 0, "HOLD": 0}
    consecutive_errors = 0
    
    try:
        while True:
            # Check MT5 connection
            if not is_mt5_connected():
                logger.warning("⚠️ MT5 connection lost!")
                if not reconnect_mt5():
                    logger.error("❌ Failed to reconnect after all attempts. Exiting.")
                    break
                consecutive_errors = 0
            
            # Wait for new bar (with connection check)
            current_bar_time = wait_for_new_bar(last_bar_time)
            
            # If None returned, connection was lost
            if current_bar_time is None:
                logger.warning("⚠️ Connection issue detected, attempting reconnect...")
                if not reconnect_mt5():
                    logger.error("❌ Failed to reconnect. Exiting.")
                    break
                consecutive_errors = 0
                continue
            
            if last_bar_time is not None and current_bar_time == last_bar_time:
                # Heartbeat
                if time.time() - last_heartbeat > Config.HEARTBEAT_INTERVAL:
                    print(f"⏳ Waiting for new bar... (Last: {last_bar_time.strftime('%H:%M')})")
                    last_heartbeat = time.time()
                time.sleep(Config.LOOP_INTERVAL_SECONDS)
                continue
            
            last_bar_time = current_bar_time
            
            # Fetch data with error handling
            try:
                df_m5 = get_mt5_data(Config.SYMBOL, Config.TIMEFRAME_FAST, Config.BARS_M5)
                df_h1 = get_mt5_data(Config.SYMBOL, Config.TIMEFRAME_SLOW, Config.BARS_H1)
                
                if df_m5 is None or df_h1 is None:
                    consecutive_errors += 1
                    logger.warning(f"⚠️ Data fetch failed ({consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS})")
                    
                    if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                        logger.warning("🔄 Too many errors, attempting reconnect...")
                        if not reconnect_mt5():
                            break
                        consecutive_errors = 0
                    
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Data fetch error: {e} ({consecutive_errors}/{Config.MAX_CONSECUTIVE_ERRORS})")
                
                if consecutive_errors >= Config.MAX_CONSECUTIVE_ERRORS:
                    if not reconnect_mt5():
                        break
                    consecutive_errors = 0
                
                time.sleep(Config.ERROR_COOLDOWN_SECONDS)
                continue
            
            # Reset error counter on success
            consecutive_errors = 0
            current_price = df_m5['close'].iloc[-1]
            
            # Run prediction
            try:
                result = adapter.predict(df_m5, df_h1)
            except Exception as e:
                logger.error(f"❌ Prediction error: {e}")
                continue
            
            # Evaluate signal
            action, emoji = evaluate_signal(result)
            
            # Print prediction
            print_prediction(
                timestamp=current_bar_time,
                close_price=current_price,
                result=result,
                action=action,
                emoji=emoji
            )
            
            # Log signal
            log_signal(current_bar_time, current_price, result, action)
            
            # === ONLINE LEARNING ===
            # 1. Add current prediction to buffer
            learning_buffer.add_sample(
                timestamp=current_bar_time,
                close_price=current_price,
                signal=result.signal,
                confidence=result.confidence,
                probabilities=result.probabilities,
            )
            
            # 2. Update outcomes for old samples
            updated = learning_buffer.update_outcomes(current_price)
            
            # 3. Check if we should retrain
            if learning_buffer.should_retrain():
                stats = learning_buffer.get_training_stats()
                logger.info(f"🎓 Online Learning: {stats['total_samples']} samples, "
                          f"Live accuracy: {stats['accuracy']:.1%}")
                learning_buffer.reset_retrain_counter()
            # === END ONLINE LEARNING ===
            
            # Track signals
            if action in ["BUY", "STRONG_BUY"]:
                signals_today["BUY"] += 1
            elif action in ["SELL", "STRONG_SELL"]:
                signals_today["SELL"] += 1
            else:
                signals_today["HOLD"] += 1
            
            # Execute trade (disabled by default)
            if action in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                execute_trade(action, current_price)
            
            # Session stats
            total = sum(signals_today.values())
            if total % 10 == 0 and total > 0:
                logger.info(f"📊 Session stats: BUY={signals_today['BUY']}, "
                          f"SELL={signals_today['SELL']}, HOLD={signals_today['HOLD']}")
            
            last_heartbeat = time.time()
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("🛑 SHUTDOWN REQUESTED")
        print("=" * 70)
        print(f"   Session signals: BUY={signals_today['BUY']}, "
              f"SELL={signals_today['SELL']}, HOLD={signals_today['HOLD']}")
        print(f"   Log file: {Config.LOG_FILE}")
        print("=" * 70 + "\n")
    
    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed. Goodbye! 👋")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_live_bot()

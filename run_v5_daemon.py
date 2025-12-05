#!/usr/bin/env python3
"""
🛡️ Golden Breeze v5 Ultimate - Daemon Watchdog
===============================================

Auto-restart wrapper for live_v5_ultimate.py
Handles:
- Automatic restart on crash
- Restart on MT5 disconnect
- Restart on any exception
- Logging of all restarts

Usage:
    python run_v5_daemon.py
    
Or run in background:
    Start-Process python -ArgumentList "run_v5_daemon.py" -WindowStyle Hidden
"""

import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Configuration
SCRIPT_PATH = Path(__file__).parent / "live_v5_ultimate.py"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Restart settings
MAX_RESTARTS_PER_HOUR = 10
RESTART_DELAY_SECONDS = 5
COOLDOWN_AFTER_MANY_RESTARTS = 300  # 5 minutes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | DAEMON | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "daemon_watchdog.log")
    ]
)
logger = logging.getLogger("Daemon")


def run_bot():
    """Run the bot and return exit code."""
    logger.info(f"🚀 Starting bot: {SCRIPT_PATH}")
    
    try:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        return process.returncode
        
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        return -1


def main():
    """Main daemon loop with auto-restart."""
    
    print("\n" + "=" * 70)
    print("🛡️  GOLDEN BREEZE v5 ULTIMATE - DAEMON WATCHDOG")
    print("=" * 70)
    print("   Auto-restart enabled")
    print(f"   Max restarts per hour: {MAX_RESTARTS_PER_HOUR}")
    print(f"   Restart delay: {RESTART_DELAY_SECONDS}s")
    print("   Press Ctrl+C to stop daemon")
    print("=" * 70 + "\n")
    
    restart_times = []
    restart_count = 0
    
    while True:
        try:
            # Run the bot
            start_time = datetime.now()
            exit_code = run_bot()
            end_time = datetime.now()
            
            runtime = (end_time - start_time).total_seconds()
            
            # Log the exit
            if exit_code == 0:
                logger.info(f"✅ Bot exited normally (runtime: {runtime:.1f}s)")
            elif exit_code == -1:
                logger.error(f"❌ Bot failed to start")
            else:
                logger.warning(f"⚠️ Bot crashed with code {exit_code} (runtime: {runtime:.1f}s)")
            
            # Track restarts
            restart_times.append(datetime.now())
            restart_count += 1
            
            # Clean old restart times (keep only last hour)
            one_hour_ago = datetime.now().timestamp() - 3600
            restart_times = [t for t in restart_times if t.timestamp() > one_hour_ago]
            
            # Check restart limit
            if len(restart_times) >= MAX_RESTARTS_PER_HOUR:
                logger.warning(f"🔥 Too many restarts ({len(restart_times)}/hour)! Cooling down for {COOLDOWN_AFTER_MANY_RESTARTS}s...")
                time.sleep(COOLDOWN_AFTER_MANY_RESTARTS)
                restart_times.clear()
            
            # Normal restart delay
            logger.info(f"🔄 Restarting in {RESTART_DELAY_SECONDS}s... (restart #{restart_count})")
            time.sleep(RESTART_DELAY_SECONDS)
            
        except KeyboardInterrupt:
            print("\n")
            logger.info("🛑 Daemon stopped by user (Ctrl+C)")
            print("\n" + "=" * 70)
            print(f"   Total restarts this session: {restart_count}")
            print("=" * 70 + "\n")
            break
            
        except Exception as e:
            logger.error(f"❌ Daemon error: {e}")
            time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    main()

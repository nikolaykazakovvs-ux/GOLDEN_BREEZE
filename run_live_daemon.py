"""
Auto-restart wrapper for Live Trading Engine.
Keeps the bot running even if it crashes or stops.
"""

import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

# Configuration
SCRIPT = "strategy/live_v4.py"
ARGS = ["--lot", "0.01", "--buy-threshold", "0.50", "--sell-threshold", "0.50"]
MAX_RESTARTS = 999999  # Unlimited restarts
RESTART_DELAY = 5  # Seconds between restarts

def log(message):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    """Main loop with auto-restart."""
    restart_count = 0
    
    log("=" * 60)
    log("🤖 Golden Breeze - Auto-Restart Daemon")
    log("=" * 60)
    log(f"Script: {SCRIPT}")
    log(f"Args: {' '.join(ARGS)}")
    log(f"Max restarts: {MAX_RESTARTS}")
    log("=" * 60)
    
    while restart_count < MAX_RESTARTS:
        restart_count += 1
        log(f"🚀 Starting live trading (attempt #{restart_count})...")
        
        try:
            # Run the trading script as a subprocess
            # Use Popen for non-blocking execution
            process = subprocess.Popen(
                [sys.executable, SCRIPT] + ARGS,
                cwd=Path(__file__).parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()  # Wait for process to complete
            
            exit_code = process.returncode
            
            if exit_code == 0:
                log("✅ Process ended normally (exit code 0)")
                break  # Normal exit, don't restart
            else:
                log(f"⚠️ Process crashed (exit code {exit_code})")
        
        except KeyboardInterrupt:
            log("⚠️ Interrupted by user (Ctrl+C)")
            break
        
        except Exception as e:
            log(f"❌ Error: {e}")
        
        # Wait before restart
        if restart_count < MAX_RESTARTS:
            log(f"⏳ Waiting {RESTART_DELAY}s before restart...")
            time.sleep(RESTART_DELAY)
    
    log("=" * 60)
    log(f"🛑 Daemon stopped after {restart_count} restart(s)")
    log("=" * 60)

if __name__ == "__main__":
    main()

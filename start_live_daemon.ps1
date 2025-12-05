# Golden Breeze - Auto-Restart Live Trading
# Runs in background, auto-restarts on crash

$ErrorActionPreference = "Stop"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "🤖 Starting Golden Breeze Live Trading Daemon" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "Process will run in BACKGROUND" -ForegroundColor Yellow
Write-Host "To stop: Close this terminal or press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Change to project directory
Set-Location "F:\Development of trading bots\Golden Breeze"

# Activate venv if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
}

# Start the daemon (it will auto-restart the trading bot)
Write-Host "🚀 Launching daemon..." -ForegroundColor Green
Write-Host ""

python run_live_daemon.py

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "✅ Daemon stopped" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Golden Breeze V4 - Live Bot Launcher
# 24/7 Paper Trading Mode

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Golden Breeze V4 - Live Bot" -ForegroundColor Cyan  
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Mode: PAPER (Demo Account)" -ForegroundColor White
Write-Host "  Model: best_long_run.pt (Epoch 21, MCC +0.282)" -ForegroundColor White
Write-Host "  Trading: 24/7" -ForegroundColor White
Write-Host ""
Write-Host "Starting bot..." -ForegroundColor Green
Write-Host ""

cd "f:\Development of trading bots\Golden Breeze"

python strategy/live_v4.py --paper --model models/v4_6year/best_long_run.pt

Write-Host ""
Write-Host "Bot stopped." -ForegroundColor Yellow
Write-Host "Press any key to close..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

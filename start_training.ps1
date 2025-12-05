# Golden Breeze V4 - Training Launcher
# Batch 512 (max from benchmark), 2 workers, 500 epochs

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  Golden Breeze V4 - Training Start" -ForegroundColor Cyan  
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Config:" -ForegroundColor Yellow
Write-Host "  Dataset: v4_6year_dataset.npz (490k samples)" -ForegroundColor White
Write-Host "  Batch Size: 24576 (INSANE + Grad Accum 2x = 49152!)" -ForegroundColor Green
Write-Host "  Workers: 0 (data on GPU - no need for CPU workers!)" -ForegroundColor White
Write-Host "  Epochs: 500" -ForegroundColor White
Write-Host "  Patience: 50" -ForegroundColor White
Write-Host "  Memory: FULL DATASET ON GPU + SHARED VRAM!" -ForegroundColor Magenta
Write-Host "  VRAM: 8 GB dedicated + 14 GB shared = 22 GB!" -ForegroundColor Cyan
Write-Host "  Effective Batch: 49152 (gradient accumulation!)" -ForegroundColor Magenta
Write-Host "  Training time: ~1 hour" -ForegroundColor Green
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

cd "f:\Development of trading bots\Golden Breeze"

python -m aimodule.training.train_v4_lstm `
    --data-path data/prepared/v4_6year_dataset.npz `
    --batch-size 24576 `
    --epochs 500 `
    --patience 50 `
    --save-dir models/v4_6year `
    --num-workers 0 `
    --preload-gpu

Write-Host ""
Write-Host "Training finished or stopped." -ForegroundColor Yellow
Write-Host "Press any key to close..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

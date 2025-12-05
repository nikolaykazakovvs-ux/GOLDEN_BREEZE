# Golden Breeze V4 - 6-Year Training Script (Optimized)
# Benchmark Results: Batch 400, FP32, 4 workers
# Expected: ~1.4 hours for 500 epochs

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Golden Breeze V4 - Long Training Run (Optimized)" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host " 📊 Dataset: v4_6year_dataset.npz (490,383 samples)" -ForegroundColor Green
Write-Host " 🎯 Target: 500 epochs, patience=50" -ForegroundColor Green
Write-Host " ⚙️ Batch Size: 400 (benchmark optimal)" -ForegroundColor Green
Write-Host " 🔥 Precision: FP32 (faster than FP16 on small LSTM)" -ForegroundColor Green
Write-Host " 🧠 Workers: 4 (Ryzen 2700 has 8 cores)" -ForegroundColor Green
Write-Host " 💾 Memory: mmap mode (prevents OOM)" -ForegroundColor Green
Write-Host ""
Write-Host " ⏱️ Expected Time: ~1.4 hours" -ForegroundColor Magenta
Write-Host " 🎯 Expected MCC: +0.30 to +0.35" -ForegroundColor Magenta
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$StartTime = Get-Date

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Run training with logging
python -m aimodule.training.train_v4_lstm `
    --data-path data/prepared/v4_6year_dataset.npz `
    --epochs 500 `
    --batch-size 400 `
    --patience 50 `
    --save-dir models/v4_6year `
    2>&1 | Tee-Object -FilePath "logs/training_v4_6year_night.log"

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " 🏁 Training Complete!" -ForegroundColor Green
Write-Host " ⏱️ Duration: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host " 📁 Model saved to: models/v4_6year/" -ForegroundColor Yellow
Write-Host " 📄 Report: models/v4_6year/training_report.json" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# run_full_training_gpu.ps1
# One-command GPU training & smoke test for Golden Breeze

param(
    [string]$Symbol = "XAUUSD",
    [string]$Start = "",
    [string]$End = "",
    [int]$Epochs = 5,
    [int]$SeqLen = 50,
    [string]$PrimaryTf = "M5"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 Golden Breeze - Full GPU Training & Smoke Test" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

# Activate venv
Write-Host "[1/6] Activating venv..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# GPU check
Write-Host "[2/6] Checking CUDA availability..." -ForegroundColor Green
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())"
if ($cudaAvailable -ne "True") {
    Write-Host "❌ CUDA not available inside venv. Please install PyTorch with CUDA and retry." -ForegroundColor Red
    exit 2
}
Write-Host "✅ CUDA available = True" -ForegroundColor Green

# Start AI Core in background
Write-Host "[3/6] Starting AI Core (FastAPI)..." -ForegroundColor Green
$aiCore = Start-Process -FilePath python -ArgumentList "-m","aimodule.server.local_ai_gateway" -PassThru -WindowStyle Minimized
Write-Host "   PID: $($aiCore.Id)" -ForegroundColor DarkGray

# Resolve Start/End default (last 10 days)
if ([string]::IsNullOrWhiteSpace($Start) -or [string]::IsNullOrWhiteSpace($End)) {
    $endDate = Get-Date
    $startDate = $endDate.AddDays(-10)
    $Start = $startDate.ToString("yyyy-MM-dd")
    $End = $endDate.ToString("yyyy-MM-dd")
}
Write-Host "   Interval: $Start → $End" -ForegroundColor DarkGray

# Wait for /health (device=cuda)
Write-Host "[4/6] Waiting for GPU health (device=cuda)..." -ForegroundColor Green
$deadline = (Get-Date).AddSeconds(60)
$healthOk = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = python -c "import requests; import json; r=requests.get('http://127.0.0.1:5005/health',timeout=2); print(json.dumps(r.json()))"
        if ($LASTEXITCODE -eq 0) {
            $obj = $resp | ConvertFrom-Json
            if ($obj.device -eq "cuda" -and $obj.use_gpu -eq $true) {
                Write-Host "✅ Health OK: device=cuda, use_gpu=true" -ForegroundColor Green
                $healthOk = $true
                break
            } else {
                Write-Host "   Waiting... device=$($obj.device), use_gpu=$($obj.use_gpu)" -ForegroundColor DarkGray
            }
        }
    } catch {}
    Start-Sleep -Seconds 2
}

if (-not $healthOk) {
    Write-Host "❌ Health timeout after 60s (device!=cuda). Stopping AI Core." -ForegroundColor Red
    try { Stop-Process -Id $aiCore.Id -Force } catch {}
    exit 3
}

# Run training pipeline
Write-Host "[5/6] Running training & backtest pipeline..." -ForegroundColor Green
$cmdArgs = @(
    "-m","tools.train_and_backtest_hybrid",
    "--symbol", $Symbol,
    "--start", $Start,
    "--end", $End,
    "--primary-tf", $PrimaryTf,
    "--seq-len", "$SeqLen",
    "--epochs", "$Epochs"
)
python $cmdArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Pipeline failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    try { Stop-Process -Id $aiCore.Id -Force } catch {}
    exit $LASTEXITCODE
}

# Stop AI Core
Write-Host "[6/6] Stopping AI Core..." -ForegroundColor Green
try {
    # Prefer graceful stop via port owner, fallback to PID
    $conns = Get-NetTCPConnection -LocalPort 5005 -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($c in $conns) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }
    } else {
        Stop-Process -Id $aiCore.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "✅ AI Core stopped" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not stop AI Core (maybe already exited)." -ForegroundColor Yellow
}

# Summarize artifacts
Write-Host "\n============================================================" -ForegroundColor Cyan
Write-Host "🎉 Training + backtest completed successfully" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

# Locate artifacts
$report = Get-ChildItem -Path .\reports -Filter "hybrid_v1.1_${Symbol}_*.md" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$model = Get-ChildItem -Path .\models -Filter "direction_lstm_hybrid_${Symbol}.pt" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($report) { Write-Host "Report: $($report.FullName)" -ForegroundColor Green } else { Write-Host "Report: not found" -ForegroundColor Red }
if ($model) { Write-Host "Model:  $($model.FullName)" -ForegroundColor Green } else { Write-Host "Model:  not found" -ForegroundColor Red }
Write-Host "Device: cuda" -ForegroundColor Green

exit 0

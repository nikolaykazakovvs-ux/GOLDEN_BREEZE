# run_server.ps1
# Скрипт для быстрого запуска AI-сервера Golden Breeze

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  GOLDEN BREEZE - AICore_XAUUSD_v3.0 (GPU-Ready)" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Активация виртуального окружения
Write-Host "[1/2] Активация виртуального окружения..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "[2/2] Запуск FastAPI сервера..." -ForegroundColor Green
Write-Host ""
Write-Host "Сервер будет доступен по адресу: " -NoNewline
Write-Host "http://127.0.0.1:5005" -ForegroundColor Cyan
Write-Host ""
Write-Host "Нажмите CTRL+C для остановки сервера" -ForegroundColor Yellow
Write-Host ""

# Запуск сервера
uvicorn aimodule.server.local_ai_gateway:app --host 127.0.0.1 --port 5005 --reload

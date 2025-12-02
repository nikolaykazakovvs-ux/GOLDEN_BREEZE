# run_test.ps1
# Скрипт для запуска тестов

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  Тестирование GOLDEN BREEZE API" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Активация виртуального окружения
Write-Host "Активация виртуального окружения..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "Запуск тестов..." -ForegroundColor Green
Write-Host ""

# Запуск тестов
python test_api.py

Write-Host ""
Write-Host "Тестирование завершено!" -ForegroundColor Green

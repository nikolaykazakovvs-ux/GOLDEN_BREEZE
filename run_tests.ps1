# run_tests.ps1
# Скрипт запуска тестов для Golden Breeze AI Core

Set-Location "$PSScriptRoot"

Write-Host "=== Golden Breeze AI Core - Running Tests ===" -ForegroundColor Cyan

# Проверка venv
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Виртуальное окружение не найдено!" -ForegroundColor Red
    Write-Host "Запустите сначала: .\run_install_ai.ps1" -ForegroundColor Yellow
    exit 1
}

# Активация
.\venv\Scripts\Activate.ps1

Write-Host "Запуск тестов с pytest..." -ForegroundColor Yellow
pytest test_ai_core.py -v --tb=short

Write-Host ""
Write-Host "=== Тестирование завершено ===" -ForegroundColor Green

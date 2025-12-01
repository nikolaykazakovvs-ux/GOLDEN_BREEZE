# run_install_ai.ps1
# Скрипт установки всех AI-зависимостей для Golden Breeze

Set-Location "$PSScriptRoot"

Write-Host "=== Golden Breeze AI Setup ===" -ForegroundColor Cyan
Write-Host "Активация виртуального окружения..." -ForegroundColor Yellow

# Проверка наличия venv
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Виртуальное окружение не найдено. Создание venv..." -ForegroundColor Yellow
    python -m venv venv
}

# Активация
.\venv\Scripts\Activate.ps1

Write-Host "Установка зависимостей из requirements.txt..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Установка завершена ===" -ForegroundColor Green
Write-Host "Для запуска сервера используйте: python -m aimodule.server.local_ai_gateway" -ForegroundColor Cyan

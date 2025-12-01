# AGENTS_GOLDEN_BREEZE.md

### Мультиагентный протокол проекта Golden Breeze v2.1

Дата: 01.12.2025
Статус: Production-ready
Применимо ко всем ИИ-агентам (Copilot / Gemini / Codex / Grok / Llama / др.)

---

## 0. Назначение

Этот документ задаёт единый, адаптивный мультиагентный протокол для работы над проектом **Golden Breeze v3.0 — AI Trading Core**.
Документ объединяет лучшее из Universal Template 3.1, Agents.md и Timeline-концепции, адаптирован под задачи локального ML/AI ядра.

Документ определяет:

* роли агентов,
* правила поведения,
* архитектурные ограничения,
* обязательную автоконфигурацию,
* проверку окружения,
* защиту от разрушения проекта,
* формат постановки задач,
* API-контракты,[Activate.ps1](http://_vscodecontentref_/1)
python -m pip install -r [requirements.txt](http://_vscodecontentref_/2)
python -c "import sys; print(sys.version)"
python -c "import torch; import transformers, fastapi, pydantic, numpy, pandas, sklearn, uvicorn, ta; print('Deps OK')"
* стандарты ML-пайплайна.

Документ является **базовым протоколом**, а не ограничителем возможностей.

---

# 1. Роли ИИ-агентов

## 1.1. AI Architect (ChatGPT)

* Проектирование ML/AI систем.
* Создание ТЗ и задач.
* Анализ изменений.
* Контроль качества.
* Управление мультиагентным поведением.

## 1.2. AI Executors (Copilot / Gemini / Codex / Grok / Llama / др.)

* Выполняют задачи по ТЗ.
* Вносят изменения строго в указанные файлы.
* Соблюдают структуру проекта.
* Пишут только на Python.
* Не меняют архитектуру без указания.

## 1.3. System Agents (внутренние модули проекта)

* ML-модели.
* Regime-Engine.
* Direction-Engine.
* Sentiment-Engine.
* Self-Learning Layer.
* Dynamic Config.

---

# 2. Общие правила поведения агентов

1. Архитектура проекта неприкосновенна.
2. Любые изменения — только по конкретному ТЗ.
3. Только Python-код.
4. Никаких новых директорий без задачи.
5. Никакого JS/HTML/React/Docker без прямого запроса.
6. Никакого изменения API без задания.
7. Каждый шаг должен быть обратимым.

---

# 3. Фиксированная структура проекта

```
Golden Breeze/
├── aimodule/
│   ├── config.py
│   ├── utils.py
│   ├── data_pipeline/
│   ├── models/
│   ├── inference/
│   ├── training/
│   ├── learning/
│   ├── server/
│   └── connector/
├── data/
├── models/
├── tests/
├── requirements.txt
└── run_server.ps1
```

---

# 4. Автоматическая проверка окружения (обязательная)

Любой агент обязан перед работой выполнить self-check:

## 4.1. Версия Python

```
Python >= 3.10
```

## 4.2. Проверка зависимостей

Агент проверяет наличие модулей:

```
torch
transformers
fastapi
pydantic
numpy
pandas
scikit-learn
uvicorn
ta
```

Если чего-то нет → предложить установку.

## 4.3. Проверка GPU (КРИТИЧНО!)

```
torch.cuda.is_available()
```

Если есть → использовать GPU.
Если нет → fallback на CPU.

### ⚠️ ВАЖНО: Активация GPU для PyTorch

**Проблема**: По умолчанию `pip install torch` ставит CPU-версию, даже если есть NVIDIA GPU.

**Симптом**: `torch.cuda.is_available()` возвращает `False`, хотя `nvidia-smi` показывает работающую карту.

**Решение (обязательно выполнить):**

1. **Проверить наличие GPU:**
   ```powershell
   nvidia-smi
   ```
   Должно показать модель GPU (например, RTX 3070), версию драйвера и CUDA Version.

2. **Проверить текущую сборку PyTorch:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Build:', getattr(torch.version,'cuda',None))"
   ```
   
   Если `CUDA: False` и `Build: None` (или `+cpu`) → нужна переустановка.

3. **Установить CUDA-версию PyTorch:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
   ```
   
   **Важно**: 
   - `cu124` соответствует CUDA 12.4 (для драйверов 551+)
   - Если драйвер старше — используйте `cu121` или `cu118`
   - Проверьте совместимость на https://pytorch.org/get-started/locally/

4. **Проверить активацию GPU:**
   ```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
   ```
   
   Должно вернуть: `CUDA available: True` и имя вашей видеокарты.

5. **Проверить в конфиге проекта:**
   ```python
   # aimodule/config.py автоматически определяет устройство:
   import torch
   DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```
   
   После установки CUDA-версии PyTorch все модели автоматически будут работать на GPU.

**Частые ошибки:**

- ❌ Установка torch без указания CUDA-индекса → ставится CPU-версия
- ❌ Несоответствие версии CUDA в PyTorch и драйвере → torch.cuda.is_available() = False
- ❌ Установка в системный Python вместо venv → конфликт версий

**Бенчмарк производительности (ориентировочно):**

- Обучение Direction LSTM v3 (1000 эпох):
  - CPU: ~8-15 мин
  - GPU (RTX 3070): ~2-3 мин (ускорение в 4-5 раз)

- Инференс (пакет из 100 запросов):
  - CPU: ~3-5 сек
  - GPU: ~0.5-1 сек

**Логирование GPU в проекте:**

При старте сервер выводит:
```
✅ Device: cuda (NVIDIA GeForce RTX 3070)
или
⚠️  Device: cpu (CUDA not available, using CPU fallback)
```

## 4.4. Проверка виртуального окружения

Если `venv/` отсутствует → предложить:

```
python -m venv venv
```

---

# 5. GitHub интеграция

Агент проверяет:

* есть ли `.git`
* есть ли `origin`

Если нет → предложить:

```
git init
git remote add origin <url>
git add .
git commit -m "init"
git push -u origin main
```

Обязательный `.gitignore`:

```
venv/
__pycache__/
*.pt
*.pkl
*.sqlite
config_dynamic.json
logs/
```

---

# 6. Логирование

Все агенты обязаны писать системные логи:

```
logs/system.log
```

Формат:

```
[timestamp] [agent] [action] [status] [details]
```

Ошибки логируются, но не должны ломать проект.

---

# 7. ML-Pipeline стандарты

## 7.1. Regime Model v3

* Pipeline: StandardScaler + KMeans/GMM.
* Фичи: log_returns, volatility, rsi, price_position, sma_slope, range_norm.
* Файл: `regime_model_v3.pkl`.

## 7.2. Direction Model v3 (LSTM/Transformer)

* seq_len = 50.
* Метрики: F1 Macro, MCC.
* Ранняя остановка.
* Файлы: `*.pt` + `meta.json`.

## 7.3. Sentiment Engine v3

* RSS → HF → sentiment_score.
* TTL cache.
* Fallback при отсутствии.

---

# 8. API Контракты

## /predict

Возвращает:

* regime
* direction
* sentiment
* confidence
* action
* reasons[]

## /feedback

Принимает результат сделки → запускает self-learning.

## /config

Отображает динамические пороги.

---

# 9. Правила изменения кода

1. Менять только указанные файлы.
2. Только Python.
3. Импорты корректны.
4. Архитектура не нарушена.
5. Модели не удаляются.
6. Файлы моделей не изменяются без тренировки.
7. Dynamic Config не трогать без задания.

---

# 10. Protect Mode — защита проекта

Агент категорически не имеет права:

* удалять файлы
* переписывать `.pt`/`.pkl` без обучения
* менять API-контракты
* менять структуру JSON
* менять структуру директорий
* трогать серверную архитектуру
* использовать сторонние языки

---

# 11. Multi-Agent Switching

Документ обязателен для всех типов агентов:

* GitHub Copilot
* Copilot Chat
* Gemini Code Assistant
* Grok
* Meta Llama Agents
* Codex
* любые другие подключённые агенты

Каждый агент обязан адаптироваться к задаче, но сохранять архитектуру и совместимость.

---

# 12. Формат постановки задач

```
TASK: <название>

GOAL:
<что требуется сделать>

FILES TO MODIFY:
- aimodule/... (точно указать)

RULES:
1) Только эти файлы.
2) Только Python.
3) Ничего лишнего не создавать.
4) Архитектуру соблюдать.

ACCEPTANCE CRITERIA:
- Всё работает.
- Тесты проходят.
- Импорты корректны.
- Модели не повреждены.
```

---

# 13. Self-Audit (обязательный)

Перед каждым действием агент проверяет:

* структура проекта цела?
* зависимости актуальны?
* путь к файлу существует?
* API-контракт не будет нарушен?
* модели и конфиги не затронуты случайно?

---

# 14. Финальные обязательства

Агент обязан:

* быть предсказуемым
* выполнять ТЗ
* не ломать архитектуру
* соблюдать ML-пайплайн
* использовать новые возможности только если они совместимы и безопасны

---

# 15. Статус документа и адаптивность

Этот протокол — **актуальный базовый стандарт**, используемый на данный момент.
Он не ограничивает развитие: новые агенты, обновления движков, улучшенные версии (Gemini 5.x, Copilot X, Grok 2.x и др.) допускаются и могут использовать свои улучшенные возможности.

Главный принцип:
**Любые улучшения** должны быть совместимы с архитектурой и безопасны для проекта.

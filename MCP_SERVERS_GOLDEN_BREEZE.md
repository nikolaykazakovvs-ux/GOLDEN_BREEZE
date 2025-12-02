# MCP_SERVERS_GOLDEN_BREEZE.md  
### Multi-Agent / MCP Server Map for Golden Breeze

Документ описывает, **какие MCP-серверы нужны проекту Golden Breeze**, зачем они нужны и какие минимальные требования к их API и поведению.

Golden Breeze = **локальное AI-ядро**, MCP-серверы = **инструменты, которыми агенты пользуются вокруг ядра** (код, данные, рынок, метрики).

---

## 0. Classification Overview

Мы делим MCP-серверы на три уровня:

- **CORE** — базовая инфраструктура разработки (без них агенты слепые).
- **TRADING** — доступ к рынку, сделкам, новостям, метрикам.
- **OPTIONAL** — удобство, эксплуатация, CI/CD, логирование.

Краткий список:

### CORE
1. `core.fs` — Project File System Server  
2. `core.git` — Git / GitHub Server  
3. `core.shell` — Shell / Process Runner Server  
4. `core.python` — Python Runtime Server  

### TRADING
5. `trading.market_data` — Market Data / OHLCV Server  
6. `trading.trade_history` — Trade History / Broker Bridge Server  
7. `trading.news` — News / Sentiment Source Server  
8. `trading.metrics` — Trading Metrics / Monitoring Server  

### OPTIONAL
9. `ops.logs` — Logs / Observability Server  
10. `ops.config` — Config Management Server  
11. `ops.cicd` — CI/CD Orchestrator Server  

---

## 1. CORE Servers

### 1.1. `core.fs` — Project File System MCP

**Purpose**  
Дать агентам безопасный доступ к файловой структуре проекта Golden Breeze.

**Scope**  
- Чтение кода: `aimodule/`, `tests/`, `models/`, `data/`, `AGENTS_*.md`, `README.md`.  
- Ограниченная запись (по ТЗ) в:
  - `aimodule/` (код)
  - `tests/` (тесты)
  - `docs/` (документация, если есть)

**Minimal API (пример)**  
- `list(path)` → список файлов/папок  
- `read_file(path)` → текст  
- `write_file(path, content)` → запись (только разрешённые зоны)  
- `exists(path)` → bool  

**Notes**  
- Никакого доступа выше корня репозитория.  
- Обязан соблюдать правила из `AGENTS_GOLDEN_BREEZE.md`.

---

### 1.2. `core.git` — Git / GitHub MCP

**Purpose**  
Управление версионностью проекта и интеграция с GitHub.

**Scope**  
- Локальный git: `status`, `diff`, `commit`.  
- Remote: `push`, `pull`, `PR` (если подключён GitHub).

**Minimal API (пример)**  
- `git_status()`  
- `git_diff(files?)`  
- `git_commit(message)`  
- `git_push(branch)`  
- `git_create_pr(title, body, branch)` (если есть GitHub)

**Notes**  
- Агенты не коммитят автоматически без явного ТЗ.  
- Используется для код-ревью, history и релизов.

---

### 1.3. `core.shell` — Shell / Process Runner MCP

**Purpose**  
Даёт агентам возможность запускать команды в проекте.

**Scope**  
- Запуск тестов: `pytest`, отдельных файлов.  
- Запуск сервера: `uvicorn ...`.  
- Установка зависимостей: `pip install ...`.

**Minimal API (пример)**  
- `run(command: str, cwd: str)` → {stdout, stderr, exit_code}

**Notes**  
- Важен жёсткий sandbox: никакого удаления файлов, работы с системными каталогами и т.п.  
- Основной сценарий: `pytest`, `uvicorn`, `python -m ...`.

---

### 1.4. `core.python` — Python Runtime MCP

**Purpose**  
Позволяет выполнять малые Python-фрагменты для проверки кода.

**Scope**  
- Проверить импорты: `import aimodule...`.  
- Создать модель и прогнать `forward`.  
- Проверить форму данных.

**Minimal API (пример)**  
- `python_exec(code: str)` → stdout / результат / traceback

**Notes**  
- Используется для “быстрых sanity-check”.  
- Не заменяет полноценные тесты.

---

## 2. TRADING Servers

### 2.1. `trading.market_data` — Market Data / OHLCV MCP

**Purpose**  
Доступ к историческим и, опционально, live данным по XAUUSD (и другим инструментам).

**Scope**  
- Исторические свечи `OHLCV` по:
  - `symbol` (например, `XAUUSD` / `XAUUSD.X`)  
  - `timeframe` (M1, M5, M15, H1, H4, D1)  
  - диапазон дат  

**Minimal API (пример)**  
- `get_ohlcv(symbol, timeframe, start, end)` → DataFrame/JSON: `[time, open, high, low, close, volume]`  

**Notes**  
- Это источник данных для:
  - обучения Regime/Direction  
  - бэктестов  
  - обновления моделей  

---

### 2.2. `trading.trade_history` — Trade History / Broker Bridge MCP

**Purpose**  
Доступ к историям сделок и, при необходимости, текущим позициям.

**Scope**  
- История сделок (closed trades).  
- Прибыль/убыток, SL/TP, время входа/выхода.  
- (Опционально) открытые позиции.

**Minimal API (пример)**  
- `get_closed_trades(account_id, symbol, start, end)` → список сделок  
- `get_open_positions(account_id)`  

**Notes**  
- Используется:
  - для заполнения `FeedbackStore`,  
  - для self-learning анализа,  
  - для отчётов и анализа режимов.

---

### 2.3. `trading.news` — News / Sentiment Source MCP

**Purpose**  
Источник новостных текстов для `Sentiment Engine v3`.

**Scope**  
- Выдаёт список коротких новостей / заголовков по `symbol` и интервалу времени.

**Minimal API (пример)**  
- `get_news(symbol, limit=10, since=None)` → `List[str]`  

**Notes**  
- MCP-сервер отвечает за:
  - RSS / News API / кастомные источники  
  - кэширование, фильтрацию  
- `Sentiment Engine` видит только текст, а не знает, откуда он взялся.

---

### 2.4. `trading.metrics` — Trading Metrics / Monitoring MCP

**Purpose**  
Доступ к агрегированным метрикам: PnL, DD, win-rate, статистика по режимам.

**Scope**  
- Equity-кривая по аккаунту.  
- PnL по дням/неделям.  
- Win-rate по режимам и символам.  
- Кол-во сделок, средний риск/профит.

**Minimal API (пример)**  
- `get_equity_curve(account_id, start, end)`  
- `get_regime_stats(account_id, symbol, regime)`  
- `get_overall_metrics(account_id)`  

**Notes**  
- На базе этого MCP можно строить:
  - аналитику,  
  - auto-recs по настройкам,  
  - алерты.

---

## 3. OPTIONAL / OPS Servers

### 3.1. `ops.logs` — Logs / Observability MCP

**Purpose**  
Чтение логов системы и ошибок ядра.

**Scope**  
- Логи:
  - `logs/system.log`  
  - лог API-ошибок  
  - лог self-learning апдейтов  

**Minimal API (пример)**  
- `get_logs(source, since, level)`  

**Notes**  
- Используется агентами для диагностики и пост-морем анализа.

---

### 3.2. `ops.config` — Config Management MCP

**Purpose**  
Безопасное управление конфигами:

- `config.py` (базовый)  
- `config_dynamic.json` (динамический)  
- отдельные файлы стратегий  

**Minimal API (пример)**  
- `get_config(name)` → JSON  
- `update_config(name, patch)` → новое состояние  

**Notes**  
- Любые изменения должны:
  - быть маленькими (patch),  
  - логироваться,  
  - по хорошему — подтверждаться человеком.  

---

### 3.3. `ops.cicd` — CI/CD Orchestrator MCP

**Purpose**  
Интеграция с CI-pipeline (GitHub Actions / GitLab CI / др.).

**Scope**  
- Запуск билдов/тестов.  
- Проверка статуса workflow.  
- Привязка к релизам Golden Breeze.

**Minimal API (пример)**  
- `trigger_pipeline(name, branch)`  
- `get_pipeline_status(id)`  

**Notes**  
- Это “будущее”, когда Golden Breeze станет продуктом с релизами.

---

## 4. Status Table (Draft)

| ID                   | Name                       | Layer    | Status       | Comment                          |
|----------------------|---------------------------|----------|-------------|----------------------------------|
| `core.fs`            | Project FS                | CORE     | planned      | Нужен в первую очередь           |
| `core.git`           | Git / GitHub              | CORE     | planned      | Для PR и версий                  |
| `core.shell`         | Shell Runner              | CORE     | planned      | Для pytest/uvicorn/training      |
| `core.python`        | Python Runtime            | CORE     | planned      | Для быстрых проверок             |
| `trading.market_data`| Market Data               | TRADING  | planned      | История и обучение               |
| `trading.trade_history`| Trade History / Broker | TRADING  | planned      | Feedback и self-learning         |
| `trading.news`       | News / Sentiment          | TRADING  | planned      | Источник текстов для Sentiment   |
| `trading.metrics`    | Metrics / Monitoring      | TRADING  | planned      | Аналитика и контроль             |
| `ops.logs`           | Logs / Observability      | OPTIONAL | planned      | Для диагностики                  |
| `ops.config`         | Config Manager            | OPTIONAL | planned      | Осторожное управление конфиго    |
| `ops.cicd`           | CI/CD Orchestrator        | OPTIONAL | future       | Когда появятся релизы            |

---

## 5. How to Evolve

1. Сначала реализуем **CORE** (fs, git, shell, python).  
2. Потом подключаем **TRADING** (market_data, trade_history, news, metrics).  
3. Далее добавляем **OPS**-слой по мере роста проекта.

Golden Breeze остаётся **чистым AI-ядром**, а MCP-серверы — это его “внешний мир”, к которому агенты получают структурированный доступ.

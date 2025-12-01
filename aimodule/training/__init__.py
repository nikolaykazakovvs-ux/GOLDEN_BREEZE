# Training Scripts - Golden Breeze v2.0

Скрипты для обучения AI-моделей на исторических данных XAUUSD.

## 📁 Файлы в этой папке

### `train_direction_lstm.py`
Обучение LSTM-модели для прогноза направления цены.

**Требования:**
- Файл `data/xauusd_m5.csv` с историческими данными
- Минимум 1000 свечей (рекомендуется 10000+)

**Запуск:**
```powershell
python -m aimodule.training.train_direction_lstm
```

**Результат:**
- Сохраняет веса в `models/direction_model.pt`

---

### `train_regime_cluster.py`
Обучение кластеризатора для определения режимов рынка.

**Методы:**
- KMeans (по умолчанию)
- GMM (Gaussian Mixture Model)

**Запуск:**
```powershell
python -m aimodule.training.train_regime_cluster
```

**Результат:**
- Сохраняет модель в `models/regime_model.pt` (joblib format)

---

### `build_sentiment_lexicon.py`
Создание словаря для локального анализа настроений.

**Запуск:**
```powershell
python -m aimodule.training.build_sentiment_lexicon
```

**Результат:**
- Сохраняет лексикон в `models/sentiment_model.gguf`

---

### `prepare_data_example.py`
Пример подготовки данных в формат CSV.

**Использование:**
```powershell
python -m aimodule.training.prepare_data_example
```

---

## 🚀 Быстрый старт

1. Подготовьте данные в `data/xauusd_m5.csv`
2. Обучите все модели:

```powershell
python -m aimodule.training.train_direction_lstm
python -m aimodule.training.train_regime_cluster
python -m aimodule.training.build_sentiment_lexicon
```

3. Перезапустите сервер для загрузки новых весов

---

## 📊 Формат данных

CSV файл должен содержать колонки:
```
timestamp,open,high,low,close,volume
```

Пример:
```csv
2025-11-30T09:00:00,2640.0,2642.0,2639.0,2641.5,1000.0
2025-11-30T09:05:00,2641.5,2643.0,2640.5,2642.0,1100.0
```

---

Подробная документация в `TRAINING_GUIDE.md`

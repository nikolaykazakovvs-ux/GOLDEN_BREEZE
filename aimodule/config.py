# aimodule/config.py

"""
Конфигурация Golden Breeze v3.0.

Включает:
- Пути к моделям и данным
- GPU/CUDA настройки
- Поддерживаемые инструменты и таймфреймы
- Динамические пороги (для self-learning)
"""

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Директории
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Создание директорий при импорте
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Инструмент по умолчанию (используется для динамических путей моделей)
DEFAULT_SYMBOL = "XAUUSD"

# GPU/CUDA настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

# Пути к моделям
REGIME_MODEL_PATH = MODELS_DIR / "regime_model.pt"
REGIME_ML_MODEL_PATH = MODELS_DIR / "regime_ml.pkl"
DIRECTION_MODEL_PATH = MODELS_DIR / "direction_model.pt"
# SMC Model v1.0 (with Fair Value Gaps and Swing Points features)
DIRECTION_LSTM_MODEL_PATH = MODELS_DIR / f"direction_lstm_smc_v1.pt"
DIRECTION_LSTM_METADATA_PATH = DIRECTION_LSTM_MODEL_PATH.with_suffix(".json")
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.gguf"

# Пути к данным
HISTORY_CSV_PATH = DATA_DIR / "xauusd_history.csv"
TRADE_FEEDBACK_PATH = DATA_DIR / "trade_feedback.csv"
DYNAMIC_CONFIG_PATH = DATA_DIR / "config_dynamic.json"

# Торговые настройки
SUPPORTED_SYMBOLS = {"XAUUSD", "XAUUSD.X", "EURUSD", "BTCUSD"}
DEFAULT_TIMEFRAME = "M5"
SUPPORTED_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}

# Параметры моделей
LSTM_SEQUENCE_LENGTH = 50  # Окно для LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

REGIME_CLUSTERS = 4  # Количество кластеров для regime model

# Пороги принятия решений (могут обновляться self-learning)
MIN_CONFIDENCE_BASE = 0.25
MIN_CONFIDENCE_VOLATILE = 0.35
SENTIMENT_SKIP_THRESHOLD = -0.4
SENTIMENT_WEAK_THRESHOLD = 0.1

# Self-learning параметры
FEEDBACK_BATCH_SIZE = 100  # Обновлять статистику каждые N сделок
FEEDBACK_WINDOW_DAYS = 30  # Анализировать последние N дней

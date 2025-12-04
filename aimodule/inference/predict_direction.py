# aimodule/inference/predict_direction.py

"""
Инференс для определения направления движения цены.
Использует DirectionLSTMWrapper при наличии, иначе fallback на DirectionPredictor.
"""

import pandas as pd
from pathlib import Path
from ..models.direction_model import DirectionPredictor
from ..utils import Direction
from ..config import DIRECTION_LSTM_MODEL_PATH

# Попытка загрузить улучшенную LSTM модель
direction_lstm_model = None
try:
    lstm_path = Path(DIRECTION_LSTM_MODEL_PATH)
    if lstm_path.exists():
        from ..models.direction_lstm_model import DirectionLSTMWrapper
        direction_lstm_model = DirectionLSTMWrapper.load(str(lstm_path))
        print(f"✅ Загружена LSTM модель направления: {lstm_path.name}")
    else:
        print(f"ℹ️  Гибридная LSTM модель не найдена ({lstm_path}), используется fallback")
except Exception as e:
    print(f"⚠️  Улучшенная LSTM модель не загружена, используется fallback: {e}")

# Fallback модель
direction_predictor = DirectionPredictor()


def infer_direction(df: pd.DataFrame) -> tuple[Direction, float]:
    """
    Определение направления движения цены.
    
    Приоритет:
    1. DirectionLSTMWrapper (если обучена и загружена)
    2. DirectionPredictor (базовая LSTM или momentum fallback)
    
    Args:
        df: DataFrame с OHLCV данными
        
    Returns:
        (Direction, confidence)
    """
    # Попытка использовать улучшенную LSTM модель
    if direction_lstm_model is not None:
        try:
            direction, confidence = direction_lstm_model.predict_proba(df)
            print(f"[DEBUG predict_direction] LSTM returned: direction={direction}, confidence={confidence:.8f}")
            
            if confidence > 0.1:  # минимальная уверенность
                print(f"[DEBUG predict_direction] Using LSTM result (confidence {confidence:.8f} > 0.1)")
                return direction, confidence
            else:
                print(f"[DEBUG predict_direction] FALLBACK: confidence {confidence:.8f} <= 0.1")
        except Exception as e:
            print(f"[DEBUG predict_direction] LSTM exception: {e}")
    
    # Fallback на базовую модель
    direction, confidence = direction_predictor.predict(df)
    print(f"[DEBUG predict_direction] Fallback returned: direction={direction}, confidence={confidence:.8f}")
    return direction, confidence

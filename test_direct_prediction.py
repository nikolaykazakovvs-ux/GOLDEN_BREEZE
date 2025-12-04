"""
Прямой тест DirectionLSTMWrapper.predict_proba() без сервера.
Загружаем данные, вызываем модель и смотрим что возвращается.
"""

import pandas as pd
from pathlib import Path
from aimodule.models.direction_lstm_model import DirectionLSTMWrapper
from aimodule.config import DIRECTION_LSTM_MODEL_PATH

print("="*70)
print("DIRECT PREDICTION TEST - DirectionLSTMWrapper.predict_proba()")
print("="*70)

# 1. Загрузка модели
model_path = Path(DIRECTION_LSTM_MODEL_PATH)
print(f"\n1. Loading model from: {model_path}")
model = DirectionLSTMWrapper.load(str(model_path))
print(f"✅ Model loaded successfully")
print(f"   - seq_length: {model.seq_length}")
print(f"   - num_classes: {model.num_classes}")
print(f"   - device: {model.device}")

# 2. Загрузка данных
data_path = Path("data/raw/XAUUSD/M5.csv")
print(f"\n2. Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"✅ Data loaded: {len(df)} candles")
print(f"   Columns: {df.columns.tolist()}")

# 3. Тест на последних 100 свечах
print(f"\n3. Testing predictions on last 100 candles...")
df_test = df.tail(100).copy()

# Сделаем 5 предсказаний на разных позициях
test_positions = [50, 60, 70, 80, 90]

print("\n" + "="*70)
print("PREDICTION RESULTS:")
print("="*70)

for pos in test_positions:
    df_window = df_test.iloc[:pos+1].copy()
    
    direction, confidence = model.predict_proba(df_window)
    
    print(f"\nPosition {pos}:")
    print(f"  Direction: {direction}")
    print(f"  Confidence: {confidence:.8f}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

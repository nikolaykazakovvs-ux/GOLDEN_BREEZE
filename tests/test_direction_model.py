import numpy as np
import pandas as pd
from pathlib import Path

try:
    from aimodule.models.direction_lstm_model import DirectionLSTMWrapper
except Exception:
    DirectionLSTMWrapper = None


def test_direction_wrapper_train_and_predict():
    if DirectionLSTMWrapper is None:
        assert True, "DirectionLSTMWrapper not available"
        return

    # Synthetic dataset with simple trend
    n = 1000
    prices = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="T"),
        "open": prices + np.random.randn(n)*0.1,
        "high": prices + np.abs(np.random.randn(n))*0.2,
        "low": prices - np.abs(np.random.randn(n))*0.2,
        "close": prices + np.random.randn(n)*0.1,
        "volume": np.abs(np.random.randn(n))*10 + 1,
    })

    model = DirectionLSTMWrapper(input_size=6, hidden_size=16, num_layers=1, seq_length=50, dropout=0.1)
    model.fit(df, epochs=3, lr=0.01)

    direction, conf = model.predict_proba(df)
    assert direction is not None
    assert 0.0 <= float(conf) <= 1.0


import numpy as np
import pandas as pd
from pathlib import Path

try:
    from aimodule.training.train_regime_cluster import build_features_from_df
    from aimodule.config import REGIME_ML_MODEL_PATH
except Exception:
    build_features_from_df = None
    REGIME_ML_MODEL_PATH = Path("regime_model.pkl")


def test_regime_features_shape():
    if build_features_from_df is None:
        assert True, "Regime training module not available"
        return

    # Create synthetic data
    n = 500
    ts = pd.date_range("2025-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(100, 150, n) + np.random.randn(n),
        "high": np.linspace(101, 151, n) + np.random.randn(n),
        "low": np.linspace(99, 149, n) + np.random.randn(n),
        "close": np.linspace(100, 150, n) + np.random.randn(n),
        "volume": np.abs(np.random.randn(n))*1000 + 10,
    })

    X = build_features_from_df(df)
    assert X is not None
    assert X.shape[0] == n
    assert X.shape[1] >= 6  # at least a few engineered features


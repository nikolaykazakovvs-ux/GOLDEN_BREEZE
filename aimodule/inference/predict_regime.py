# aimodule/inference/predict_regime.py

"""
Инференс для определения режима рынка.
Использует RegimeMLModel при наличии, иначе fallback на RegimeClusterModel.
"""

import pandas as pd
from pathlib import Path
from ..data_pipeline.features import add_basic_features
from ..models.regime_model import RegimeClusterModel
from ..utils import MarketRegime
from ..config import REGIME_MODEL_PATH

# Попытка загрузить ML модель
regime_ml_model = None
try:
    # Проверяем наличие модели regime_ml.pkl
    regime_ml_path = Path(REGIME_MODEL_PATH).parent / "regime_ml.pkl"
    if regime_ml_path.exists():
        from ..models.regime_ml_model import RegimeMLModel
        regime_ml_model = RegimeMLModel.load(str(regime_ml_path))
        print(f"✅ Загружена ML-модель режима: {regime_ml_path}")
except Exception as e:
    print(f"⚠️  ML-модель режима не загружена, используется fallback: {e}")

# Fallback модель
regime_cluster_model = RegimeClusterModel()


def infer_regime(df: pd.DataFrame) -> MarketRegime:
    """
    Определение режима рынка на основе исторических данных.
    
    Приоритет:
    1. RegimeMLModel (если обучена и загружена)
    2. RegimeClusterModel (простая кластеризация)
    
    Args:
        df: DataFrame с OHLCV данными
        
    Returns:
        MarketRegime
    """
    df_feat = add_basic_features(df)
    
    # Попытка использовать ML модель
    if regime_ml_model is not None:
        try:
            regime = regime_ml_model.predict(df_feat)
            if regime != MarketRegime.UNKNOWN:
                return regime
        except Exception:
            pass
    
    # Fallback на простую модель
    regime = regime_cluster_model.predict(df_feat)
    return regime

# aimodule/training/train_regime_cluster.py

"""
Обучение кластеризатора рыночных режимов (KMeans/GMM) для Golden Breeze v3.0.

Обновления v3.0:
- Расширенный признак-ингиниринг (волатильность, лог-доходности, позиция цены в диапазоне, наклон SMA, RSI)
- Масштабирование признаков (StandardScaler)
- Выбор модели: KMeans или GMM через аргументы
- Сохранение пайплайна (скейлер + модель) в `REGIME_MODEL_PATH`
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from ..data_pipeline.loader import prepare_training_dataframe, load_history_csv
from ..config import REGIME_ML_MODEL_PATH as REGIME_MODEL_PATH, MODELS_DIR


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Строим матрицу признаков для кластеризации режимов рынка.

    Используем расширенные признаки из v3.0:
    - log_returns: лог-доходности
    - volatility: скользящая стандартная девиация
    - sma_slope: наклон быстрой SMA
    - price_position: позиция закрытия в дневном диапазоне [0..1]
    - rsi: относительная сила
    - range_norm: нормализованный внутридневной диапазон
    """
    dff = prepare_training_dataframe(df)

    log_returns = dff.get("log_returns", pd.Series([0.0]*len(dff))).astype(float).fillna(0.0)
    volatility = dff.get("volatility", pd.Series([0.0]*len(dff))).astype(float).fillna(0.0)
    sma_slope = dff.get("sma_slope", pd.Series([0.0]*len(dff))).astype(float).fillna(0.0)
    price_position = dff.get("price_position", pd.Series([0.5]*len(dff))).astype(float).fillna(0.5)
    rsi = dff.get("rsi", pd.Series([50.0]*len(dff))).astype(float).fillna(50.0)
    range_norm = ((dff["high"].astype(float) - dff["low"].astype(float)) / (dff["close"].astype(float) + 1e-8)).fillna(0.0)

    X = np.stack([
        log_returns.values,
        volatility.values,
        sma_slope.values,
        price_position.values,
        rsi.values,
        range_norm.values
    ], axis=1)
    return X


def train_kmeans(df: pd.DataFrame, n_clusters: int = 4):
    X = build_features(df)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=n_clusters, n_init=25, random_state=42))
    ])
    pipe.fit(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, REGIME_MODEL_PATH)
    print(f"Saved KMeans pipeline to {REGIME_MODEL_PATH}")


def train_gmm(df: pd.DataFrame, n_components: int = 4):
    X = build_features(df)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianMixture(n_components=n_components, random_state=42))
    ])
    pipe.fit(X)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, REGIME_MODEL_PATH)
    print(f"Saved GMM pipeline to {REGIME_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regime clustering model (v3.0)")
    parser.add_argument("--data", type=str, default=str(Path("data/xauusd_m5.csv")), help="Путь к CSV с историей")
    parser.add_argument("--model", type=str, choices=["kmeans", "gmm"], default="kmeans", help="Тип модели")
    parser.add_argument("--clusters", type=int, default=4, help="Количество кластеров/компонент")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден {data_path}. Подготовьте исторические данные.")

    # Загрузка с нормализацией временных меток и проверкой колонок
    df = load_history_csv(data_path)

    if args.model == "kmeans":
        train_kmeans(df, n_clusters=args.clusters)
    else:
        train_gmm(df, n_components=args.clusters)

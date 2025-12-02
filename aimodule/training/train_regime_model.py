# aimodule/training/train_regime_model.py

"""
Скрипт обучения ML-модели режима рынка (Market Regime Detector).

Использование:
    python -m aimodule.training.train_regime_model

Требования:
    - Файл data/xauusd_history.csv с колонками: timestamp, open, high, low, close, volume
    - Минимум 10000 строк для качественного обучения
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем корень проекта в путь
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from aimodule.models.regime_ml_model import RegimeMLModel
from aimodule.data_pipeline.features import add_basic_features
from aimodule.config import REGIME_MODEL_PATH


def prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Загрузка и подготовка данных для обучения.
    
    Args:
        csv_path: путь к CSV файлу с историей
        
    Returns:
        DataFrame с добавленными техническими индикаторами
    """
    print(f"📂 Загрузка данных из {csv_path}...")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Файл не найден: {csv_path}\n"
            f"Создайте файл с историческими данными XAUUSD в формате:\n"
            f"timestamp,open,high,low,close,volume"
        )
    
    df = pd.read_csv(csv_path)
    
    required_cols = ["timestamp", "open", "high", "low", "close"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    
    print(f"✅ Загружено {len(df)} свечей")
    
    # Добавление технических индикаторов
    print("🔧 Расчёт технических индикаторов...")
    df = add_basic_features(df)
    
    # Добавление дополнительных фич для режима
    print("🔧 Добавление дополнительных признаков...")
    
    # Returns (доходность)
    df['returns'] = df['close'].pct_change().fillna(0)
    
    # SMA slope (наклон скользящей средней)
    if 'sma_fast' in df.columns:
        df['sma_slope'] = df['sma_fast'].diff().fillna(0)
    
    # Очистка NaN
    df = df.dropna()
    
    print(f"✅ Подготовлено {len(df)} свечей с признаками")
    
    return df


def train_model(
    df: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int = 4
) -> RegimeMLModel:
    """
    Обучение модели режима рынка.
    
    Args:
        df: DataFrame с историей и признаками
        method: 'kmeans' или 'gmm'
        n_clusters: количество кластеров (режимов)
        
    Returns:
        Обученная RegimeMLModel
    """
    print(f"\n🎯 Обучение модели (method={method}, n_clusters={n_clusters})...")
    
    model = RegimeMLModel(method=method, n_clusters=n_clusters)
    
    try:
        model.fit(df)
        print("✅ Модель успешно обучена")
        
        # Статистика по кластерам
        print("\n📊 Распределение режимов:")
        features = model._extract_features(df)
        if features is not None:
            features_scaled = model.scaler.transform(features)
            labels = model.clusterer.predict(features_scaled)
            
            unique, counts = np.unique(labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                regime = model.cluster_map.get(cluster_id, "UNKNOWN")
                pct = count / len(labels) * 100
                print(f"  Кластер {cluster_id} ({regime}): {count} ({pct:.1f}%)")
        
        return model
    
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        raise


def save_model(model: RegimeMLModel, output_path: str):
    """Сохранение обученной модели."""
    print(f"\n💾 Сохранение модели в {output_path}...")
    
    try:
        model.save(output_path)
        print("✅ Модель успешно сохранена")
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {e}")
        raise


def main():
    """Основная функция обучения."""
    print("=" * 60)
    print("🚀 Golden Breeze - Обучение ML-модели режима рынка")
    print("=" * 60)
    
    # Пути
    data_path = project_root / "data" / "xauusd_history.csv"
    output_path = project_root / "models" / "regime_ml.pkl"
    
    # Подготовка данных
    try:
        df = prepare_data(str(data_path))
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\n📝 Для продолжения:")
        print("   1. Создайте папку 'data' в корне проекта")
        print("   2. Поместите туда файл 'xauusd_history.csv' с историческими данными")
        print("   3. Запустите скрипт снова")
        return
    
    # Обучение модели
    # Можно выбрать method='gmm' для GaussianMixture
    model = train_model(df, method="kmeans", n_clusters=4)
    
    # Сохранение
    save_model(model, str(output_path))
    
    print("\n" + "=" * 60)
    print("✅ Обучение завершено успешно!")
    print("=" * 60)
    print(f"\n📍 Модель сохранена: {output_path}")
    print("🔄 Перезапустите AI-сервер для загрузки новой модели:")
    print("   python -m aimodule.server.local_ai_gateway")


if __name__ == "__main__":
    main()

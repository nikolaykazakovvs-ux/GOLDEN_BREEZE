"""Training script for Direction Model v3 (LSTM).

Использует расширенные признаки и обучает многоклассовую модель направления.
Сохраняет лучшую модель по MCC и meta JSON.
"""

import argparse
from pathlib import Path
import pandas as pd

from ..data_pipeline.loader import load_history_csv, prepare_training_dataframe
from ..models.direction_lstm_model import DirectionLSTMWrapper
from ..config import DIRECTION_LSTM_MODEL_PATH


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = load_history_csv(path)
    df = df.sort_values("timestamp")
    df = prepare_training_dataframe(df)
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Train Direction LSTM Model v3")
    parser.add_argument("--data", type=str, default="data/xauusd_history.csv", help="Путь к CSV историческим данным")
    parser.add_argument("--seq-len", type=int, default=50, help="Длина окна LSTM")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=64, help="Размер батча")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Доля валидации")
    parser.add_argument("--epsilon", type=float, default=0.0005, help="Порог классификации LONG/SHORT")
    parser.add_argument("--hidden-size", type=int, default=64, help="Размер скрытого слоя")
    parser.add_argument("--num-layers", type=int, default=2, help="Число LSTM слоёв")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Не найден файл данных: {data_path}")

    print("[+] Загрузка и подготовка данных...")
    df = load_and_prepare(data_path)
    if len(df) < args.seq_len + 200:
        print(f"⚠️  Мало данных ({len(df)}), качество может быть снижено")

    print("[+] Инициализация модели...")
    wrapper = DirectionLSTMWrapper(
        seq_length=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epsilon=args.epsilon,
        seed=args.seed
    )

    print("[+] Обучение...")
    result = wrapper.fit(
        df=df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        early_stopping_patience=args.patience
    )

    print("[+] Результат:")
    print(f"    best_epoch = {result['best_epoch']}")
    print(f"    best_mcc   = {result['best_mcc']:.4f}")
    print(f"    model saved to {DIRECTION_LSTM_MODEL_PATH}")
    print(f"    meta saved to {DIRECTION_LSTM_MODEL_PATH}.meta.json")


if __name__ == "__main__":
    main()

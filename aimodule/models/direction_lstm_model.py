# aimodule/models/direction_lstm_model.py

"""
Direction Model v3 — улучшенная LSTM модель прогноза направления.

Обновления v3:
- Используются расширенные признаки из подготовленного DataFrame:
    [close, log_returns, volatility, sma_slope, price_position, rsi]
- Гибкая длина окна (sequence length)
- Классы: FLAT(0) / LONG(1) / SHORT(2) с порогом epsilon
- Train/Val split + метрики (Accuracy, F1 Macro, MCC)
- Early stopping по MCC
- Сохранение лучшей модели + метаданные .meta.json
- GPU поддержка (DEVICE из config)
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import json
import time
import random
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from ..utils import Direction
from ..config import (
    DIRECTION_LSTM_MODEL_PATH,
    DIRECTION_LSTM_METADATA_PATH,
    DEVICE,
    PROJECT_ROOT,
)


class DirectionLSTMModel(nn.Module):
    """
    Улучшенная LSTM модель для классификации направления.
    
    Поддерживает две головы:
    - two_layer: FC1 + ReLU + FC2 (старый формат)
    - single_layer: единственный FC (новый обучающий пайплайн v1.1)
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        head_type: str = "two_layer"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.head_type = head_type
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        if head_type == "single_layer":
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc1 = nn.Linear(hidden_size, max(2, hidden_size // 2))
            self.fc2 = nn.Linear(max(2, hidden_size // 2), num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        
        if self.head_type == "single_layer":
            logits = self.fc(out)
        else:
            out = self.fc1(out)
            out = self.relu(out)
            out = self.dropout(out)
            logits = self.fc2(out)
        
        return logits


class DirectionLSTMWrapper:
    """
    Обёртка для обучения и инференса DirectionLSTMModel.
    
    Методы:
    - train(df, epochs): обучение на историческом DataFrame
    - predict_proba(df): предсказание вероятностей классов
    - save(path): сохранение весов
    - load(path): загрузка весов
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_length: int = 50,
        dropout: float = 0.2,
        num_classes: int = 3,
        head_type: str = "two_layer",
        epsilon: float = 0.0005,
        seed: int = 42
    ):
        self.device = DEVICE
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.head_type = head_type
        self.epsilon = epsilon
        self.seed = seed
        self._set_seed(seed)

        self.model = DirectionLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            head_type=head_type
        ).to(self.device)

        self.scaler_mean = None
        self.scaler_std = None
        self.best_mcc = -1.0
        self.best_state: Optional[Dict[str, Any]] = None

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Извлечение признаков.

        Используем полный набор из 15 признаков (с SMC features):
        close, returns, log_returns, sma_fast, sma_slow, sma_ratio,
        atr, atr_norm, rsi, bb_position, volume_ratio,
        SMC_FVG_Bullish, SMC_FVG_Bearish, SMC_Swing_High, SMC_Swing_Low.
        
        Если признаков нет в df - подставляем безопасные значения.
        """
        if len(df) == 0:
            return None
        
        # Базовые признаки
        close = df.get("close", pd.Series([0.0]*len(df))).astype(float).ffill().fillna(0).values
        returns = df.get("returns", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        log_returns = df.get("log_returns", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        
        # SMA features
        sma_fast = df.get("sma_fast", pd.Series([0.0]*len(df))).astype(float).ffill().fillna(0).values
        sma_slow = df.get("sma_slow", pd.Series([0.0]*len(df))).astype(float).ffill().fillna(0).values
        sma_ratio = df.get("sma_ratio", pd.Series([1.0]*len(df))).astype(float).fillna(1.0).values
        
        # Volatility features
        atr = df.get("atr", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        atr_norm = df.get("atr_norm", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        
        # Indicators
        rsi = df.get("rsi", pd.Series([50.0]*len(df))).astype(float).fillna(50.0).values
        bb_position = df.get("bb_position", pd.Series([0.5]*len(df))).astype(float).fillna(0.5).values
        volume_ratio = df.get("volume_ratio", pd.Series([1.0]*len(df))).astype(float).fillna(1.0).values
        
        # SMC features
        smc_fvg_bullish = df.get("SMC_FVG_Bullish", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        smc_fvg_bearish = df.get("SMC_FVG_Bearish", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        smc_swing_high = df.get("SMC_Swing_High", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        smc_swing_low = df.get("SMC_Swing_Low", pd.Series([0.0]*len(df))).astype(float).fillna(0).values
        
        # Stack all 15 features
        features = np.stack([
            close, returns, log_returns,
            sma_fast, sma_slow, sma_ratio,
            atr, atr_norm,
            rsi,
            bb_position,
            volume_ratio,
            smc_fvg_bullish, smc_fvg_bearish,
            smc_swing_high, smc_swing_low
        ], axis=1)
        
        return features.astype(np.float32)
    
    def _normalize(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Нормализация признаков."""
        if fit:
            self.scaler_mean = np.mean(features, axis=0, keepdims=True)
            self.scaler_std = np.std(features, axis=0, keepdims=True) + 1e-8
        
        return (features - self.scaler_mean) / self.scaler_std
    
    def _create_sequences(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Создание rolling window последовательностей.
        
        Args:
            features: (n_samples, n_features)
            labels: (n_samples,) - таргеты (опционально)
            
        Returns:
            X: (n_sequences, seq_length, n_features)
            y: (n_sequences,) или None
        """
        n = len(features)
        if n < self.seq_length:
            return np.array([]), None
        
        X_list = []
        y_list = [] if labels is not None else None
        
        for i in range(self.seq_length, n):
            X_list.append(features[i - self.seq_length:i])
            if labels is not None:
                y_list.append(labels[i])
        
        X = np.array(X_list)
        y = np.array(y_list) if y_list is not None else None
        
        return X, y
    
    def _build_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Создание меток направлений по правилу epsilon.

        LONG  если close[i+1] > close[i] * (1 + epsilon)
        SHORT если close[i+1] < close[i] * (1 - epsilon)
        FLAT  иначе.
        Последняя точка не имеет будущего значения → отбрасывается.
        """
        closes = df["close"].astype(float).values
        next_closes = np.roll(closes, -1)
        labels = np.zeros(len(closes) - 1, dtype=np.int64)  # без последней
        for i in range(len(closes) - 1):
            c = closes[i]
            n = next_closes[i]
            if n > c * (1 + self.epsilon):
                labels[i] = 1  # LONG
            elif n < c * (1 - self.epsilon):
                labels[i] = 2  # SHORT
            else:
                labels[i] = 0  # FLAT
        return labels

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.2,
        early_stopping_patience: int = 5,
    ) -> Dict[str, Any]:
        """Полное обучение с валидацией и ранней остановкой.

        Returns: метрики лучшей эпохи.
        """
        features = self._extract_features(df)
        if features is None:
            raise ValueError("Не удалось извлечь признаки из DataFrame")
        labels = self._build_labels(df)
        # Обрезаем features до длины labels + 1 исходного смещения
        features = features[:-1]

        features_norm = self._normalize(features, fit=True)
        X, y = self._create_sequences(features_norm, labels)
        if len(X) == 0:
            raise ValueError(f"Недостаточно данных: нужно >= {self.seq_length + 1} строк")

        n_total = len(X)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).long().to(self.device)

        train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_epoch = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(train_loader))

            # Валидация
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()
                val_preds = np.argmax(val_probs, axis=1)
                val_loss = criterion(val_logits, y_val_t).item()

            acc = accuracy_score(y_val, val_preds)
            f1 = f1_score(y_val, val_preds, average='macro')
            try:
                mcc = matthews_corrcoef(y_val, val_preds)
            except Exception:
                mcc = 0.0

            print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f} val_loss={val_loss:.4f} acc={acc:.4f} f1={f1:.4f} mcc={mcc:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_f1": f1,
                "val_mcc": mcc
            })

            # Early stopping + сохранение лучшей модели по MCC
            if mcc > self.best_mcc:
                self.best_mcc = mcc
                self.best_state = {
                    'model_state': self.model.state_dict(),
                    'scaler_mean': self.scaler_mean,
                    'scaler_std': self.scaler_std,
                    'seq_length': self.seq_length,
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'epsilon': self.epsilon,
                    'seed': self.seed
                }
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        # Сохранение лучшей модели
        if self.best_state is not None:
            self._save_checkpoint(DIRECTION_LSTM_MODEL_PATH, self.best_state)
            self._save_metadata(DIRECTION_LSTM_MODEL_PATH, history, best_epoch)
            print(f"Saved best Direction LSTM model to {DIRECTION_LSTM_MODEL_PATH} (MCC={self.best_mcc:.4f})")

        return {
            "best_epoch": best_epoch,
            "best_mcc": self.best_mcc,
            "history": history
        }
    
    def predict_proba(self, df: pd.DataFrame) -> Tuple[Direction, float]:
        """
        Предсказание направления с вероятностью.
        
        Args:
            df: DataFrame с последними свечами (raw OHLCV)
            
        Returns:
            (Direction, confidence)
        """
        # Применяем feature engineering перед извлечением признаков
        from ..data_pipeline.features import add_basic_features
        df = add_basic_features(df.copy())
        
        features = self._extract_features(df)
        if features is None or len(features) < self.seq_length:
            return Direction.FLAT, 0.0
        
        # Нормализация
        features_norm = self._normalize(features, fit=False)
        
        # Последнее окно
        last_window = features_norm[-self.seq_length:]
        X = torch.from_numpy(last_window).unsqueeze(0).float().to(self.device)
        
        # Инференс
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        
        # FLAT=0, LONG=1, SHORT=2
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        # Debug output
        print(f"[DEBUG DirectionLSTMWrapper] probs={probs.tolist()}, confidence={confidence:.8f}, idx={idx}")
        
        if self.num_classes == 2:
            direction_map = {0: Direction.LONG, 1: Direction.SHORT}
        else:
            direction_map = {0: Direction.FLAT, 1: Direction.LONG, 2: Direction.SHORT}
        return direction_map.get(idx, Direction.FLAT), confidence
    
    def _save_checkpoint(self, path: Path, state: Dict[str, Any]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def _save_metadata(self, model_path: Path, history: list, best_epoch: int):
        meta = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_file": str(model_path),
            "seq_length": self.seq_length,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epsilon": self.epsilon,
            "device": str(self.device),
            "best_epoch": best_epoch,
            "best_mcc": self.best_mcc,
            "history": history[-10:]  # последние 10 эпох
        }
        meta_path = Path(str(model_path) + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save(self, path: Optional[str] = None):
        """Принудительное сохранение текущего состояния без метрик."""
        if path is None:
            path = DIRECTION_LSTM_MODEL_PATH
        state = {
            'model_state': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'seq_length': self.seq_length,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'epsilon': self.epsilon,
            'seed': self.seed,
            'dropout': self.dropout,
            'head_type': self.head_type
        }
        self._save_checkpoint(Path(path), state)
    
    @staticmethod
    def _resolve_metadata_path(model_path: Path) -> Path:
        if DIRECTION_LSTM_METADATA_PATH.exists():
            return DIRECTION_LSTM_METADATA_PATH
        candidate = model_path.with_suffix(".json")
        return candidate if candidate.exists() else candidate

    @staticmethod
    def _load_metadata(meta_path: Path) -> Dict[str, Any]:
        if meta_path and meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                pass
        return {}

    @staticmethod
    def _load_scalers_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        dataset_path = metadata.get("dataset_path")
        if not dataset_path:
            return None, None
        dataset_file = Path(dataset_path)
        if not dataset_file.is_absolute():
            dataset_file = (PROJECT_ROOT / dataset_file).resolve()
        if not dataset_file.exists():
            return None, None
        try:
            with np.load(dataset_file, allow_pickle=True) as data:
                mean = data["scaler_mean"] if "scaler_mean" in data.files else None
                scale_key = "scaler_scale" if "scaler_scale" in data.files else "scaler_std" if "scaler_std" in data.files else None
                scale = data[scale_key] if scale_key else None
                if mean is not None and scale is not None:
                    return np.array(mean), np.array(scale)
        except Exception:
            return None, None
        return None, None

    @classmethod
    def load(cls, path: Optional[str] = None, metadata_path: Optional[Path] = None) -> 'DirectionLSTMWrapper':
        if path is None:
            path = DIRECTION_LSTM_MODEL_PATH
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")

        checkpoint = torch.load(path, map_location='cpu')
        meta_path = metadata_path or cls._resolve_metadata_path(path)
        metadata = cls._load_metadata(meta_path)

        is_mapping = isinstance(checkpoint, dict)
        state_dict = checkpoint.get('model_state') if is_mapping else None
        if state_dict is None:
            state_dict = checkpoint

        scaler_mean = checkpoint.get('scaler_mean') if is_mapping else None
        scaler_std = None
        if is_mapping:
            scaler_std = checkpoint.get('scaler_std') or checkpoint.get('scaler_scale')

        if scaler_mean is None or scaler_std is None:
            meta_mean, meta_std = cls._load_scalers_from_metadata(metadata)
            scaler_mean = scaler_mean or meta_mean
            scaler_std = scaler_std or meta_std

        head_type = "single_layer" if any(str(k).startswith("fc.") for k in state_dict.keys()) else "two_layer"
        if any(str(k).startswith("fc1.") for k in state_dict.keys()):
            head_type = "two_layer"
        if any(str(k).startswith("fc2.") for k in state_dict.keys()) and not any(str(k).startswith("fc.") for k in state_dict.keys()):
            head_type = "two_layer"

        inferred_classes = metadata.get('n_classes') or (checkpoint.get('num_classes') if is_mapping else None)
        if inferred_classes is None:
            if head_type == "single_layer" and 'fc.bias' in state_dict:
                inferred_classes = state_dict['fc.bias'].shape[0]
            elif head_type == "two_layer" and 'fc2.bias' in state_dict:
                inferred_classes = state_dict['fc2.bias'].shape[0]
        if inferred_classes is None:
            inferred_classes = 3

        instance = cls(
            input_size=int(metadata.get('n_features', checkpoint.get('input_size', 6) if is_mapping else 6)),
            seq_length=int(metadata.get('seq_len', checkpoint.get('seq_length', 50) if is_mapping else 50)),
            hidden_size=int(metadata.get('hidden_size', checkpoint.get('hidden_size', 64) if is_mapping else 64)),
            num_layers=int(metadata.get('num_layers', checkpoint.get('num_layers', 2) if is_mapping else 2)),
            dropout=float(metadata.get('dropout', checkpoint.get('dropout', 0.2) if is_mapping else 0.2)),
            num_classes=int(inferred_classes),
            head_type=head_type,
            epsilon=float(checkpoint.get('epsilon', 0.0005) if is_mapping else 0.0005),
            seed=int(checkpoint.get('seed', 42) if is_mapping else 42)
        )

        instance.model.load_state_dict(state_dict)

        if scaler_mean is not None and scaler_std is not None:
            scaler_mean = np.array(scaler_mean, dtype=np.float32)
            scaler_std = np.array(scaler_std, dtype=np.float32)
            if scaler_mean.ndim == 1:
                scaler_mean = scaler_mean.reshape(1, -1)
            if scaler_std.ndim == 1:
                scaler_std = scaler_std.reshape(1, -1)
            instance.scaler_mean = scaler_mean
            instance.scaler_std = scaler_std
        else:
            # Безопасные значения по умолчанию
            instance.scaler_mean = np.zeros((1, instance.input_size), dtype=np.float32)
            instance.scaler_std = np.ones((1, instance.input_size), dtype=np.float32)

        instance.model.eval()
        return instance

# aimodule/learning/feedback_store.py

"""
Хранилище обратной связи о результатах торговых сделок.

Сохраняет информацию о сделках для последующего анализа и обучения.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from ..config import TRADE_FEEDBACK_PATH


class FeedbackStore:
    """
    Класс для сохранения и управления обратной связью по сделкам.
    
    Структура данных:
    - timestamp: время сделки
    - symbol: торговый инструмент
    - direction: направление сделки (long/short)
    - action: принятое действие (enter_long/enter_short/hold/skip)
    - entry_price: цена входа
    - exit_price: цена выхода (если есть)
    - pnl: прибыль/убыток
    - regime: определённый режим рынка
    - sentiment: значение sentiment
    - confidence: уверенность модели
    - actual_outcome: фактический результат (profit/loss/breakeven)
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: путь к CSV файлу (по умолчанию из config)
        """
        self.csv_path = Path(csv_path) if csv_path else TRADE_FEEDBACK_PATH
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создание файла с заголовками, если не существует
        if not self.csv_path.exists():
            self._init_csv()
    
    def _init_csv(self):
        """Создание CSV файла с заголовками."""
        columns = [
            "timestamp", "symbol", "direction", "action",
            "entry_price", "exit_price", "pnl",
            "regime", "sentiment", "confidence",
            "actual_outcome"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_path, index=False)
    
    def add_feedback(
        self,
        symbol: str,
        direction: str,
        action: str,
        regime: str,
        sentiment: float,
        confidence: float,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Добавление записи об обратной связи.
        
        Args:
            symbol: торговый инструмент (XAUUSD, EURUSD и т.д.)
            direction: направление (long, short, flat)
            action: действие (enter_long, enter_short, hold, skip)
            regime: режим рынка (trend_up, trend_down, range, volatile)
            sentiment: значение sentiment [-1, 1]
            confidence: уверенность модели [0, 1]
            entry_price: цена входа (опционально)
            exit_price: цена выхода (опционально)
            pnl: прибыль/убыток (опционально)
            timestamp: время сделки (по умолчанию текущее)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Определение фактического результата
        actual_outcome = "unknown"
        if pnl is not None:
            if pnl > 0:
                actual_outcome = "profit"
            elif pnl < 0:
                actual_outcome = "loss"
            else:
                actual_outcome = "breakeven"
        
        # Создание записи
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "direction": direction,
            "action": action,
            "entry_price": entry_price if entry_price is not None else 0.0,
            "exit_price": exit_price if exit_price is not None else 0.0,
            "pnl": pnl if pnl is not None else 0.0,
            "regime": regime,
            "sentiment": sentiment,
            "confidence": confidence,
            "actual_outcome": actual_outcome
        }
        
        # Добавление в CSV
        df = pd.DataFrame([record])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
    
    def load_feedback(
        self,
        days: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Загрузка обратной связи из CSV.
        
        Args:
            days: загрузить последние N дней (опционально)
            symbol: фильтр по инструменту (опционально)
            
        Returns:
            DataFrame с записями обратной связи
        """
        if not self.csv_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_path)
        
        if df.empty:
            return df
        
        # Парсинг timestamp
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass
        
        # Фильтр по времени
        if days is not None and "timestamp" in df.columns:
            cutoff = datetime.now() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
        
        # Фильтр по символу
        if symbol is not None:
            df = df[df["symbol"] == symbol]
        
        return df
    
    def get_statistics(
        self,
        days: Optional[int] = 30,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получение статистики по обратной связи.
        
        Args:
            days: анализировать последние N дней
            symbol: фильтр по инструменту
            
        Returns:
            Словарь со статистикой:
            - total_trades: общее количество сделок
            - profitable_trades: прибыльные сделки
            - loss_trades: убыточные сделки
            - win_rate: процент прибыльных
            - avg_pnl: средний PnL
            - by_regime: статистика по режимам
            - by_direction: статистика по направлениям
        """
        df = self.load_feedback(days=days, symbol=symbol)
        
        if df.empty:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "loss_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0
            }
        
        # Только сделки с известным результатом
        df_trades = df[df["actual_outcome"].isin(["profit", "loss", "breakeven"])]
        
        total = len(df_trades)
        profitable = len(df_trades[df_trades["actual_outcome"] == "profit"])
        losses = len(df_trades[df_trades["actual_outcome"] == "loss"])
        
        win_rate = profitable / total if total > 0 else 0.0
        avg_pnl = df_trades["pnl"].mean() if total > 0 else 0.0
        
        # Статистика по режимам
        by_regime = {}
        for regime in df_trades["regime"].unique():
            regime_df = df_trades[df_trades["regime"] == regime]
            regime_profitable = len(regime_df[regime_df["actual_outcome"] == "profit"])
            regime_total = len(regime_df)
            by_regime[regime] = {
                "total": regime_total,
                "win_rate": regime_profitable / regime_total if regime_total > 0 else 0.0,
                "avg_confidence": regime_df["confidence"].mean()
            }
        
        # Статистика по направлениям
        by_direction = {}
        for direction in df_trades["direction"].unique():
            dir_df = df_trades[df_trades["direction"] == direction]
            dir_profitable = len(dir_df[dir_df["actual_outcome"] == "profit"])
            dir_total = len(dir_df)
            by_direction[direction] = {
                "total": dir_total,
                "win_rate": dir_profitable / dir_total if dir_total > 0 else 0.0
            }
        
        return {
            "total_trades": total,
            "profitable_trades": profitable,
            "loss_trades": losses,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "by_regime": by_regime,
            "by_direction": by_direction
        }
    
    def clear_old_records(self, days: int = 90):
        """
        Удаление старых записей (старше N дней).
        
        Args:
            days: хранить последние N дней
        """
        df = self.load_feedback()
        
        if df.empty or "timestamp" not in df.columns:
            return
        
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df_filtered = df[df["timestamp"] >= cutoff]
        
        # Перезапись файла
        df_filtered.to_csv(self.csv_path, index=False)

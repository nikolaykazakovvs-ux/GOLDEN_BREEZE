# aimodule/learning/online_update.py

"""
Online обновление параметров и порогов на основе обратной связи.

Реализует легковесное online-learning:
- Обновление порогов confidence для разных режимов
- Корректировка sentiment thresholds
- Сохранение динамической конфигурации
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import (
    DYNAMIC_CONFIG_PATH,
    MIN_CONFIDENCE_BASE,
    MIN_CONFIDENCE_VOLATILE,
    SENTIMENT_SKIP_THRESHOLD,
    FEEDBACK_BATCH_SIZE
)
from .feedback_store import FeedbackStore


class OnlineUpdater:
    """
    Класс для online обновления параметров модели на основе обратной связи.
    
    Не переобучает нейросети (слишком тяжело для real-time),
    а обновляет статистические пороги и веса.
    """
    
    def __init__(
        self,
        feedback_store: Optional[FeedbackStore] = None,
        config_path: Optional[str] = None
    ):
        """
        Args:
            feedback_store: хранилище обратной связи
            config_path: путь к динамической конфигурации
        """
        self.feedback_store = feedback_store or FeedbackStore()
        self.config_path = Path(config_path) if config_path else DYNAMIC_CONFIG_PATH
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Загрузка или инициализация конфигурации
        self.dynamic_config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка динамической конфигурации из JSON."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Дефолтная конфигурация
        return {
            "min_confidence_by_regime": {
                "trend_up": MIN_CONFIDENCE_BASE,
                "trend_down": MIN_CONFIDENCE_BASE,
                "range": MIN_CONFIDENCE_BASE + 0.05,
                "volatile": MIN_CONFIDENCE_VOLATILE,
                "unknown": 0.40
            },
            "sentiment_skip_threshold": SENTIMENT_SKIP_THRESHOLD,
            "sentiment_weak_threshold": 0.1,
            "last_update_count": 0,
            "total_updates": 0
        }
    
    def _save_config(self):
        """Сохранение динамической конфигурации в JSON."""
        with open(self.config_path, "w") as f:
            json.dump(self.dynamic_config, indent=2, fp=f)
    
    def update_thresholds(self, force: bool = False) -> Dict[str, Any]:
        """
        Обновление порогов на основе накопленной обратной связи.
        
        Args:
            force: принудительное обновление (иначе только после batch_size сделок)
            
        Returns:
            Словарь с результатами обновления
        """
        # Получение статистики
        stats = self.feedback_store.get_statistics(days=30)
        
        total_trades = stats["total_trades"]
        last_update = self.dynamic_config.get("last_update_count", 0)
        
        # Проверка необходимости обновления
        if not force and (total_trades - last_update) < FEEDBACK_BATCH_SIZE:
            return {
                "updated": False,
                "reason": f"Not enough new trades ({total_trades - last_update} < {FEEDBACK_BATCH_SIZE})"
            }
        
        updates = {}
        
        # Обновление порогов по режимам
        if "by_regime" in stats:
            for regime, regime_stats in stats["by_regime"].items():
                if regime_stats["total"] < 10:
                    continue  # Недостаточно данных
                
                win_rate = regime_stats["win_rate"]
                current_threshold = self.dynamic_config["min_confidence_by_regime"].get(
                    regime, MIN_CONFIDENCE_BASE
                )
                
                # Адаптация порога
                if win_rate < 0.45:
                    # Плохая точность → повышаем порог
                    new_threshold = min(current_threshold + 0.05, 0.75)
                    updates[f"regime_{regime}_threshold"] = {
                        "old": current_threshold,
                        "new": new_threshold,
                        "reason": f"Low win rate ({win_rate:.2%})"
                    }
                    self.dynamic_config["min_confidence_by_regime"][regime] = new_threshold
                
                elif win_rate > 0.60 and current_threshold > MIN_CONFIDENCE_BASE:
                    # Хорошая точность → можем снизить порог
                    new_threshold = max(current_threshold - 0.02, MIN_CONFIDENCE_BASE)
                    updates[f"regime_{regime}_threshold"] = {
                        "old": current_threshold,
                        "new": new_threshold,
                        "reason": f"High win rate ({win_rate:.2%})"
                    }
                    self.dynamic_config["min_confidence_by_regime"][regime] = new_threshold
        
        # Обновление sentiment порогов
        if stats["total_trades"] > 50:
            # Анализ сделок с негативным sentiment
            df = self.feedback_store.load_feedback(days=30)
            if not df.empty:
                negative_sentiment = df[df["sentiment"] < 0]
                if len(negative_sentiment) > 20:
                    neg_win_rate = len(
                        negative_sentiment[negative_sentiment["actual_outcome"] == "profit"]
                    ) / len(negative_sentiment)
                    
                    current_skip = self.dynamic_config["sentiment_skip_threshold"]
                    
                    if neg_win_rate < 0.40:
                        # Негативный sentiment действительно плохой → ужесточаем
                        new_skip = max(current_skip - 0.05, -0.6)
                        updates["sentiment_skip_threshold"] = {
                            "old": current_skip,
                            "new": new_skip,
                            "reason": f"Negative sentiment has low win rate ({neg_win_rate:.2%})"
                        }
                        self.dynamic_config["sentiment_skip_threshold"] = new_skip
        
        # Обновление счётчиков
        self.dynamic_config["last_update_count"] = total_trades
        self.dynamic_config["total_updates"] += 1
        
        # Сохранение
        self._save_config()
        
        return {
            "updated": True,
            "total_trades_analyzed": total_trades,
            "updates": updates,
            "current_config": self.dynamic_config
        }
    
    def get_min_confidence(self, regime: str) -> float:
        """
        Получение минимального порога confidence для данного режима.
        
        Args:
            regime: режим рынка (trend_up, trend_down, range, volatile, unknown)
            
        Returns:
            Минимальный порог confidence
        """
        return self.dynamic_config["min_confidence_by_regime"].get(
            regime, MIN_CONFIDENCE_BASE
        )
    
    def get_sentiment_skip_threshold(self) -> float:
        """Получение порога sentiment для SKIP действия."""
        return self.dynamic_config.get("sentiment_skip_threshold", SENTIMENT_SKIP_THRESHOLD)
    
    def get_config(self) -> Dict[str, Any]:
        """Получение полной динамической конфигурации."""
        return self.dynamic_config.copy()
    
    def reset_config(self):
        """Сброс конфигурации к дефолтным значениям."""
        self.dynamic_config = self._load_config()
        if self.config_path.exists():
            self.config_path.unlink()

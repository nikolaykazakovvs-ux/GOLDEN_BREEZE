# strategy/regime_strategies.py
"""
Стратегии для разных режимов рынка:
- TrendStrategy: trend_up / trend_down
- RangeStrategy: range / flat
- VolatileStrategy: volatile / choppy
"""

from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime


class BaseRegimeStrategy(ABC):
    """Базовый класс для стратегий режимов"""
    
    def __init__(self, config, regime_name: str):
        self.config = config
        self.regime_name = regime_name
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, ai_signal: Dict) -> Optional[Dict]:
        """
        Генерация торгового сигнала
        
        Args:
            data: DataFrame с OHLCV + индикаторами
            ai_signal: Сигнал от AI Core
            
        Returns:
            Signal dict или None
        """
        pass
    
    @abstractmethod
    def calculate_sl_tp(self, entry_price: float, direction: str, 
                       atr: float) -> Tuple[float, float]:
        """
        Расчёт SL и TP
        
        Returns:
            (sl_price, tp_price)
        """
        pass
    
    def validate_signal(self, signal: Dict, ai_signal: Dict) -> bool:
        """Валидация сигнала перед входом"""
        # Проверка confidence
        if ai_signal.get("direction_confidence", 0) < self.config.min_direction_confidence:
            return False
        
        # Проверка sentiment
        if ai_signal.get("sentiment", 0) < self.config.min_sentiment_threshold:
            return False
        
        return True


class TrendStrategy(BaseRegimeStrategy):
    """
    Стратегия для трендового рынка (trend_up / trend_down)
    
    Логика:
    - Breakout entries (пробой уровней)
    - Только в направлении тренда
    - Частичные TP + trailing stop
    """
    
    def __init__(self, config):
        super().__init__(config, "trend")
    
    def generate_signal(self, data: pd.DataFrame, ai_signal: Dict) -> Optional[Dict]:
        """
        Генерация сигнала для тренда
        
        Условия:
        - regime = trend_up или trend_down
        - direction совпадает с трендом
        - Пробой локального уровня
        """
        regime = ai_signal.get("regime", "unknown")
        direction = ai_signal.get("direction", "flat")
        confidence = ai_signal.get("direction_confidence", 0)
        
        # Только trend_up или trend_down
        if regime not in ["trend_up", "trend_down"]:
            return None
        
        # Проверка направления
        if regime == "trend_up" and direction != "long":
            return None
        if regime == "trend_down" and direction != "short":
            return None
        
        # Валидация
        if not self.validate_signal({}, ai_signal):
            return None
        
        # Получаем последнюю свечу
        last_candle = data.iloc[-1]
        current_price = last_candle["close"]
        atr = last_candle.get("atr", 100.0)
        
        # Определение уровня для пробоя
        lookback = 20
        if direction == "long":
            # Ищем локальный максимум для пробоя вверх
            breakout_level = data["high"].iloc[-lookback:].max()
            
            # Если текущая цена близка к пробою
            if current_price >= breakout_level * 0.998:  # 0.2% от уровня
                sl, tp = self.calculate_sl_tp(current_price, "long", atr)
                
                return {
                    "type": "buy_stop",
                    "price": breakout_level + 5,  # Чуть выше уровня
                    "sl": sl,
                    "tp": tp,
                    "reason": f"Trend breakout up @ {breakout_level:.2f}",
                    "confidence": confidence,
                    "regime": regime
                }
        
        else:  # short
            # Ищем локальный минимум для пробоя вниз
            breakout_level = data["low"].iloc[-lookback:].min()
            
            if current_price <= breakout_level * 1.002:
                sl, tp = self.calculate_sl_tp(current_price, "short", atr)
                
                return {
                    "type": "sell_stop",
                    "price": breakout_level - 5,  # Чуть ниже уровня
                    "sl": sl,
                    "tp": tp,
                    "reason": f"Trend breakout down @ {breakout_level:.2f}",
                    "confidence": confidence,
                    "regime": regime
                }
        
        return None
    
    def calculate_sl_tp(self, entry_price: float, direction: str, 
                       atr: float) -> Tuple[float, float]:
        """
        SL и TP для тренда
        
        Trend mode:
        - SL = ATR * 2.0
        - TP = ATR * 4.0 (но с частичной фиксацией на ATR * 2.0)
        """
        sl_distance = atr * self.config.default_sl_atr_mult
        tp_distance = atr * self.config.default_tp_atr_mult
        
        if direction == "long":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return (sl, tp)
    
    def should_trail_stop(self, position: Dict, current_price: float) -> bool:
        """
        Проверка: нужно ли включать трейлинг
        
        Условие: прибыль >= min_profit_for_trail (в R)
        """
        entry = position["entry"]
        sl = position["sl"]
        
        risk = abs(entry - sl)
        current_profit = abs(current_price - entry)
        
        profit_in_r = current_profit / risk if risk > 0 else 0
        
        return profit_in_r >= self.config.trend_min_profit_for_trail
    
    def calculate_trailing_distance(self, atr: float) -> float:
        """Расстояние для трейлинг-стопа"""
        return atr * self.config.trend_trailing_atr_mult


class RangeStrategy(BaseRegimeStrategy):
    """
    Стратегия для флэта (range)
    
    Логика:
    - Mean reversion (отскоки от границ)
    - Лимитные ордера на границах диапазона
    - Фиксированный TP, жёсткий SL
    """
    
    def __init__(self, config):
        super().__init__(config, "range")
    
    def generate_signal(self, data: pd.DataFrame, ai_signal: Dict) -> Optional[Dict]:
        """
        Сигнал для range
        
        Условия:
        - regime = range
        - RSI в зонах перекупленности/перепроданности
        - Цена у границы диапазона
        """
        regime = ai_signal.get("regime", "unknown")
        
        if regime != "range":
            return None
        
        # Валидация
        if not self.validate_signal({}, ai_signal):
            return None
        
        last_candle = data.iloc[-1]
        current_price = last_candle["close"]
        atr = last_candle.get("atr", 100.0)
        
        # Проверка ATR (слишком высокая волатильность = не range)
        if atr > self.config.range_max_atr_threshold:
            return None
        
        # Определение границ диапазона
        lookback = 50
        range_high = data["high"].iloc[-lookback:].max()
        range_low = data["low"].iloc[-lookback:].min()
        range_middle = (range_high + range_low) / 2
        
        # RSI (если есть)
        rsi = last_candle.get("rsi", 50)
        
        # Long от нижней границы
        if current_price <= range_low * 1.01 and rsi < self.config.range_rsi_oversold:
            sl, tp = self.calculate_sl_tp(current_price, "long", atr, range_low)
            
            return {
                "type": "buy_limit",
                "price": range_low + 10,  # Чуть выше нижней границы
                "sl": sl,
                "tp": tp,
                "reason": f"Range bounce from {range_low:.2f}, RSI={rsi:.1f}",
                "confidence": ai_signal.get("direction_confidence", 0.5),
                "regime": regime
            }
        
        # Short от верхней границы
        elif current_price >= range_high * 0.99 and rsi > self.config.range_rsi_overbought:
            sl, tp = self.calculate_sl_tp(current_price, "short", atr, range_high)
            
            return {
                "type": "sell_limit",
                "price": range_high - 10,  # Чуть ниже верхней границы
                "sl": sl,
                "tp": tp,
                "reason": f"Range bounce from {range_high:.2f}, RSI={rsi:.1f}",
                "confidence": ai_signal.get("direction_confidence", 0.5),
                "regime": regime
            }
        
        return None
    
    def calculate_sl_tp(self, entry_price: float, direction: str, 
                       atr: float, boundary: float) -> Tuple[float, float]:
        """
        SL и TP для range
        
        Range mode:
        - SL = за границей диапазона
        - TP = фиксированный (range_tp_fixed_points)
        """
        tp_distance = self.config.range_tp_fixed_points
        
        if direction == "long":
            sl = boundary - atr  # SL ниже нижней границы
            tp = entry_price + tp_distance
        else:
            sl = boundary + atr  # SL выше верхней границы
            tp = entry_price - tp_distance
        
        return (sl, tp)


class VolatileStrategy(BaseRegimeStrategy):
    """
    Стратегия для volatile (высокая волатильность)
    
    Логика по умолчанию: NO TRADE
    Опционально: только очень уверенные breakout-сделки
    """
    
    def __init__(self, config):
        super().__init__(config, "volatile")
    
    def generate_signal(self, data: pd.DataFrame, ai_signal: Dict) -> Optional[Dict]:
        """
        Сигнал для volatile
        
        По умолчанию: None (NO TRADE)
        
        Если allow_trades = True:
        - Только при очень высоком confidence (> 0.8)
        - Уменьшенный риск
        """
        regime = ai_signal.get("regime", "unknown")
        
        if regime != "volatile":
            return None
        
        # Если запрещены сделки в volatile
        if not self.config.volatile_allow_trades:
            return None
        
        direction = ai_signal.get("direction", "flat")
        confidence = ai_signal.get("direction_confidence", 0)
        
        # Очень высокий порог confidence
        if confidence < self.config.volatile_min_confidence:
            return None
        
        # Валидация
        if not self.validate_signal({}, ai_signal):
            return None
        
        last_candle = data.iloc[-1]
        current_price = last_candle["close"]
        atr = last_candle.get("atr", 100.0)
        
        # Простой breakout по направлению AI
        if direction == "long":
            sl, tp = self.calculate_sl_tp(current_price, "long", atr)
            
            return {
                "type": "buy_stop",
                "price": current_price + 20,
                "sl": sl,
                "tp": tp,
                "reason": f"Volatile breakout long, confidence={confidence:.2f}",
                "confidence": confidence,
                "regime": regime,
                "risk_reduction": self.config.volatile_risk_reduction
            }
        
        elif direction == "short":
            sl, tp = self.calculate_sl_tp(current_price, "short", atr)
            
            return {
                "type": "sell_stop",
                "price": current_price - 20,
                "sl": sl,
                "tp": tp,
                "reason": f"Volatile breakout short, confidence={confidence:.2f}",
                "confidence": confidence,
                "regime": regime,
                "risk_reduction": self.config.volatile_risk_reduction
            }
        
        return None
    
    def calculate_sl_tp(self, entry_price: float, direction: str, 
                       atr: float) -> Tuple[float, float]:
        """
        SL и TP для volatile
        
        Wider stops из-за высокой волатильности
        """
        sl_distance = atr * (self.config.default_sl_atr_mult * 1.5)  # +50% к SL
        tp_distance = atr * (self.config.default_tp_atr_mult * 1.5)
        
        if direction == "long":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return (sl, tp)

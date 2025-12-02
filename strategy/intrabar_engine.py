# strategy/intrabar_engine.py
"""
Интрабарный движок для работы с тиками и M1 данными внутри M5 свечей
"""

from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class Tick:
    """Тиковые данные"""
    timestamp: datetime
    bid: float
    ask: float
    volume: float = 0.0
    
    @property
    def mid(self) -> float:
        """Средняя цена"""
        return (self.bid + self.ask) / 2.0
    
    @property
    def spread(self) -> float:
        """Спред в пунктах"""
        return self.ask - self.bid


@dataclass
class IntrabarCandle:
    """M1 свеча для интрабарного движения"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_ticks(self, num_ticks: int = 10) -> List[Tick]:
        """
        Генерация тиков из M1 свечи для симуляции
        
        Последовательность: Open → High → Low → Close
        """
        ticks = []
        spread = 0.5  # Примерный спред для XAUUSD
        
        # 1. Open
        ticks.append(Tick(
            timestamp=self.timestamp,
            bid=self.open - spread/2,
            ask=self.open + spread/2,
            volume=self.volume / num_ticks
        ))
        
        # 2. Move to High
        for i in range(1, num_ticks // 3):
            price = self.open + (self.high - self.open) * (i / (num_ticks // 3))
            ticks.append(Tick(
                timestamp=self.timestamp,
                bid=price - spread/2,
                ask=price + spread/2,
                volume=self.volume / num_ticks
            ))
        
        # 3. Move to Low
        for i in range(num_ticks // 3, 2 * num_ticks // 3):
            ratio = (i - num_ticks // 3) / (num_ticks // 3)
            price = self.high + (self.low - self.high) * ratio
            ticks.append(Tick(
                timestamp=self.timestamp,
                bid=price - spread/2,
                ask=price + spread/2,
                volume=self.volume / num_ticks
            ))
        
        # 4. Move to Close
        for i in range(2 * num_ticks // 3, num_ticks):
            ratio = (i - 2 * num_ticks // 3) / (num_ticks // 3)
            price = self.low + (self.close - self.low) * ratio
            ticks.append(Tick(
                timestamp=self.timestamp,
                bid=price - spread/2,
                ask=price + spread/2,
                volume=self.volume / num_ticks
            ))
        
        return ticks


class IntrabarEngine:
    """
    Движок для обработки интрабарных событий
    
    Умеет:
    - Работать с тиками MT5
    - Симулировать тики из M1 данных
    - Проверять триггеры (пробои, достижения уровней)
    - Обрабатывать SL/TP/Trailing Stop внутри свечи
    """
    
    def __init__(self, config):
        self.config = config
        self.triggers: Dict[str, List[Callable]] = {
            "price_above": [],
            "price_below": [],
            "price_cross_up": [],
            "price_cross_down": []
        }
        
    def add_trigger(self, trigger_type: str, level: float, callback: Callable):
        """
        Добавить триггер на достижение уровня
        
        Args:
            trigger_type: "price_above", "price_below", "price_cross_up", "price_cross_down"
            level: Уровень цены
            callback: Функция, которая вызывается при срабатывании
        """
        if trigger_type not in self.triggers:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
        
        self.triggers[trigger_type].append({
            "level": level,
            "callback": callback,
            "active": True
        })
    
    def process_tick(self, tick: Tick, prev_tick: Optional[Tick] = None) -> List[Dict]:
        """
        Обработка одного тика и проверка всех триггеров
        
        Returns:
            List of triggered events
        """
        events = []
        current_price = tick.mid
        prev_price = prev_tick.mid if prev_tick else current_price
        
        # Проверка price_above
        for trigger in self.triggers["price_above"]:
            if trigger["active"] and current_price >= trigger["level"]:
                events.append({
                    "type": "price_above",
                    "level": trigger["level"],
                    "tick": tick
                })
                trigger["callback"](tick)
                trigger["active"] = False
        
        # Проверка price_below
        for trigger in self.triggers["price_below"]:
            if trigger["active"] and current_price <= trigger["level"]:
                events.append({
                    "type": "price_below",
                    "level": trigger["level"],
                    "tick": tick
                })
                trigger["callback"](tick)
                trigger["active"] = False
        
        # Проверка price_cross_up (пересечение снизу вверх)
        for trigger in self.triggers["price_cross_up"]:
            if trigger["active"] and prev_price < trigger["level"] <= current_price:
                events.append({
                    "type": "price_cross_up",
                    "level": trigger["level"],
                    "tick": tick
                })
                trigger["callback"](tick)
                trigger["active"] = False
        
        # Проверка price_cross_down (пересечение сверху вниз)
        for trigger in self.triggers["price_cross_down"]:
            if trigger["active"] and prev_price > trigger["level"] >= current_price:
                events.append({
                    "type": "price_cross_down",
                    "level": trigger["level"],
                    "tick": tick
                })
                trigger["callback"](tick)
                trigger["active"] = False
        
        return events
    
    def process_m1_candle(self, candle: IntrabarCandle) -> List[Dict]:
        """
        Обработка M1 свечи как последовательности тиков
        """
        ticks = candle.to_ticks(num_ticks=10)
        all_events = []
        
        prev_tick = None
        for tick in ticks:
            events = self.process_tick(tick, prev_tick)
            all_events.extend(events)
            prev_tick = tick
        
        return all_events
    
    def clear_triggers(self):
        """Очистка всех триггеров"""
        for trigger_type in self.triggers:
            self.triggers[trigger_type] = []
    
    def check_stop_loss(self, position: Dict, tick: Tick) -> bool:
        """
        Проверка достижения Stop Loss
        
        Args:
            position: {"type": "long/short", "entry": price, "sl": price}
            tick: Текущий тик
            
        Returns:
            True если SL сработал
        """
        if position["type"] == "long":
            return tick.bid <= position["sl"]
        else:  # short
            return tick.ask >= position["sl"]
    
    def check_take_profit(self, position: Dict, tick: Tick) -> bool:
        """Проверка достижения Take Profit"""
        if position["type"] == "long":
            return tick.bid >= position["tp"]
        else:  # short
            return tick.ask <= position["tp"]
    
    def update_trailing_stop(self, position: Dict, tick: Tick, 
                            trailing_distance: float) -> Optional[float]:
        """
        Обновление трейлинг-стопа
        
        Args:
            position: Позиция
            tick: Текущий тик
            trailing_distance: Расстояние трейлинга
            
        Returns:
            Новое значение SL или None
        """
        current_price = tick.bid if position["type"] == "long" else tick.ask
        
        if position["type"] == "long":
            # Long: трейлинг поднимается
            new_sl = current_price - trailing_distance
            if new_sl > position["sl"]:
                return new_sl
        else:
            # Short: трейлинг опускается
            new_sl = current_price + trailing_distance
            if new_sl < position["sl"]:
                return new_sl
        
        return None
    
    def simulate_order_execution(self, order: Dict, tick: Tick) -> Optional[Dict]:
        """
        Симуляция исполнения ордера
        
        Args:
            order: {"type": "buy_stop/sell_stop/buy_limit/sell_limit", 
                   "price": level, "sl": sl, "tp": tp, "volume": lots}
            tick: Текущий тик
            
        Returns:
            Filled position или None
        """
        order_type = order["type"]
        order_price = order["price"]
        
        filled = False
        execution_price = None
        
        if order_type == "buy_stop":
            # Buy Stop: покупка выше текущей цены
            if tick.ask >= order_price:
                filled = True
                execution_price = max(tick.ask, order_price + self.config.slippage_points)
        
        elif order_type == "sell_stop":
            # Sell Stop: продажа ниже текущей цены
            if tick.bid <= order_price:
                filled = True
                execution_price = min(tick.bid, order_price - self.config.slippage_points)
        
        elif order_type == "buy_limit":
            # Buy Limit: покупка ниже текущей цены
            if tick.ask <= order_price:
                filled = True
                execution_price = min(tick.ask, order_price)
        
        elif order_type == "sell_limit":
            # Sell Limit: продажа выше текущей цены
            if tick.bid >= order_price:
                filled = True
                execution_price = max(tick.bid, order_price)
        
        if filled:
            return {
                "type": "long" if "buy" in order_type else "short",
                "entry": execution_price,
                "sl": order["sl"],
                "tp": order["tp"],
                "volume": order["volume"],
                "timestamp": tick.timestamp
            }
        
        return None

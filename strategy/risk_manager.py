# strategy/risk_manager.py
"""
Risk Manager для строгого контроля рисков
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd


@dataclass
class Trade:
    """Информация о сделке"""
    id: str
    symbol: str
    direction: str  # "long" / "short"
    entry_price: float
    entry_time: datetime
    volume: float
    sl: float
    tp: float
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    regime: str = "unknown"
    reason: str = ""
    

class RiskManager:
    """
    Менеджер рисков с жёсткими лимитами
    
    Контролирует:
    - Риск на сделку
    - Дневную просадку
    - Общую просадку
    - Количество позиций
    - Торговые сессии
    """
    
    def __init__(self, config, initial_balance: float):
        self.config = config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Активные позиции
        self.open_positions: Dict[str, Trade] = {}
        
        # История сделок
        self.trade_history: List[Trade] = []
        
        # Дневная статистика
        self.daily_pnl: Dict[date, float] = {}
        self.daily_trades: Dict[date, int] = {}
        
        # Лимиты достигнуты
        self.daily_limit_reached = False
        self.total_limit_reached = False
    
    def can_open_position(self, timestamp: datetime) -> Tuple[bool, str]:
        """
        Проверка: можно ли открыть новую позицию
        
        Returns:
            (allowed, reason)
        """
        # Проверка торговой сессии
        if not self._is_trading_session(timestamp):
            return False, "Outside trading session"
        
        # Проверка количества позиций
        if len(self.open_positions) >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"
        
        # Проверка дневного лимита
        if self.daily_limit_reached:
            return False, "Daily loss limit reached"
        
        # Проверка общего лимита
        if self.total_limit_reached:
            return False, "Total drawdown limit reached"
        
        return True, "OK"
    
    def calculate_position_size(self, entry_price: float, sl_price: float, 
                               direction: str, risk_reduction: float = 1.0) -> float:
        """
        Расчёт размера позиции по риску
        
        Args:
            entry_price: Цена входа
            sl_price: Цена стоп-лосса
            direction: "long" / "short"
            risk_reduction: Множитель для уменьшения риска (например, 0.5 для volatile)
            
        Returns:
            Объём позиции в лотах
        """
        # Риск на сделку в долларах
        risk_amount = self.current_balance * (self.config.risk_per_trade_pct / 100.0)
        risk_amount *= risk_reduction  # Применяем корректировку
        
        # Расстояние до SL в пунктах
        sl_distance = abs(entry_price - sl_price)
        
        # Стоимость пункта (для XAUUSD: $1 за 0.01, лот = 1 унция)
        # Примерное значение: 1 лот XAUUSD = $1 за пункт
        point_value = 1.0  # Настраивается под брокера
        
        # Объём = риск / (расстояние до SL * стоимость пункта)
        volume = risk_amount / (sl_distance * point_value) if sl_distance > 0 else 0.01
        
        # Ограничения
        volume = max(0.01, min(volume, 10.0))  # От 0.01 до 10 лотов
        
        return round(volume, 2)
    
    def open_position(self, trade: Trade) -> str:
        """
        Открытие позиции
        
        Returns:
            trade_id
        """
        trade.id = f"T{len(self.trade_history) + 1:05d}"
        self.open_positions[trade.id] = trade
        
        # Обновление дневной статистики
        trade_date = trade.entry_time.date()
        self.daily_trades[trade_date] = self.daily_trades.get(trade_date, 0) + 1
        
        return trade.id
    
    def close_position(self, trade_id: str, exit_price: float, 
                      exit_time: datetime) -> Optional[Trade]:
        """
        Закрытие позиции
        
        Returns:
            Закрытая сделка
        """
        if trade_id not in self.open_positions:
            return None
        
        trade = self.open_positions.pop(trade_id)
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        
        # Расчёт PnL
        if trade.direction == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.volume
        
        # Обновление баланса
        self.current_balance += trade.pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        # Обновление дневного PnL
        trade_date = exit_time.date()
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0) + trade.pnl
        
        # Проверка лимитов
        self._check_limits(trade_date)
        
        # Добавление в историю
        self.trade_history.append(trade)
        
        return trade
    
    def _check_limits(self, current_date: date):
        """Проверка достижения лимитов"""
        # Дневная просадка
        daily_pnl = self.daily_pnl.get(current_date, 0)
        daily_loss_pct = (daily_pnl / self.initial_balance) * 100
        
        if daily_loss_pct <= -self.config.max_daily_loss_pct:
            self.daily_limit_reached = True
        
        # Общая просадка
        current_dd = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
        
        if current_dd >= self.config.max_total_dd_pct:
            self.total_limit_reached = True
    
    def reset_daily_limits(self):
        """Сброс дневных лимитов (вызывается в начале нового дня)"""
        self.daily_limit_reached = False
    
    def _is_trading_session(self, timestamp: datetime) -> bool:
        """Проверка торговой сессии"""
        hour_utc = timestamp.hour
        return self.config.session_start_utc <= hour_utc < self.config.session_end_utc
    
    def get_current_drawdown(self) -> float:
        """Текущая просадка в %"""
        if self.peak_balance == 0:
            return 0.0
        return ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
    
    def get_daily_pnl(self, target_date: date) -> float:
        """PnL за конкретный день"""
        return self.daily_pnl.get(target_date, 0.0)
    
    def get_statistics(self) -> Dict:
        """Статистика по сделкам"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "current_dd_pct": self.get_current_drawdown()
            }
        
        wins = [t for t in self.trade_history if t.pnl and t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl and t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in self.trade_history if t.pnl)
        
        return {
            "total_trades": len(self.trade_history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trade_history) * 100 if self.trade_history else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(self.trade_history) if self.trade_history else 0,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "current_dd_pct": self.get_current_drawdown()
        }
    
    def get_regime_statistics(self) -> Dict[str, Dict]:
        """Статистика по режимам рынка"""
        regime_stats = {}
        
        for trade in self.trade_history:
            if trade.regime not in regime_stats:
                regime_stats[trade.regime] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0
                }
            
            regime_stats[trade.regime]["trades"] += 1
            if trade.pnl and trade.pnl > 0:
                regime_stats[trade.regime]["wins"] += 1
            if trade.pnl:
                regime_stats[trade.regime]["total_pnl"] += trade.pnl
        
        # Добавляем win rate
        for regime in regime_stats:
            stats = regime_stats[regime]
            stats["win_rate"] = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        
        return regime_stats

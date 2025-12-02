# strategy/config.py
"""
Конфигурация гибридной стратегии
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class StrategyConfig:
    """Конфигурация Golden Breeze Hybrid Strategy"""
    
    # === Базовые параметры ===
    symbol: str = "XAUUSD"
    
    # === Мультитаймфреймовая логика ===
    primary_tf: str = "M5"  # Динамически изменяемый рабочий таймфрейм
    supported_timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "H1", "H4"])
    execution_tf: str = "M1"  # Младший TF для интрабарного исполнения
    context_tf_high: str = "H1"  # Старший TF для контекста
    
    # Настройки Timeframe Selector
    tf_selector_min_confidence: float = 0.65  # Мин confidence для работы на TF
    tf_selector_high_confidence: float = 0.8  # Высокая confidence (усиленный приоритет)
    tf_selector_enable: bool = True  # Включить автоматический выбор PRIMARY_TF
    
    # Устаревшие параметры (для обратной совместимости)
    base_timeframe: str = "M5"  # Deprecated, использовать primary_tf
    intrabar_timeframe: str = "M1"  # Deprecated, использовать execution_tf
    
    # === Торговая сессия ===
    session_start_utc: int = 2  # 02:00 UTC
    session_end_utc: int = 22   # 22:00 UTC
    
    # === Risk Management ===
    risk_per_trade_pct: float = 1.0  # % от депозита на сделку
    max_daily_loss_pct: float = 3.0  # Макс дневная просадка
    max_total_dd_pct: float = 10.0   # Макс общая просадка
    max_positions: int = 3            # Макс одновременных позиций
    max_bars_hold: int = 100          # Макс баров в позиции (особенно для range)
    
    # === AI Core Integration ===
    ai_api_url: str = "http://127.0.0.1:5005"
    min_direction_confidence: float = 0.65  # Мин confidence для входа
    min_sentiment_threshold: float = -0.2   # Мин sentiment
    
    # === Trend Mode Settings ===
    trend_partial_tp_pct: float = 50.0  # % позиции для частичного TP
    trend_trailing_atr_mult: float = 2.0  # ATR multiplier для трейлинга
    trend_min_profit_for_trail: float = 0.5  # Мин профит для включения трейлинга (в R)
    
    # === Range Mode Settings ===
    range_tp_fixed_points: float = 100.0  # Фикс TP в пунктах для range
    range_max_atr_threshold: float = 150.0  # Макс ATR для range режима
    range_rsi_oversold: float = 30.0
    range_rsi_overbought: float = 70.0
    
    # === Volatile Mode Settings ===
    volatile_risk_reduction: float = 0.5  # Уменьшение риска в volatile (0.5 = 50%)
    volatile_min_confidence: float = 0.8  # Высокий порог для входа
    volatile_allow_trades: bool = False   # По умолчанию NO TRADE в volatile
    
    # === Stop Loss / Take Profit ===
    default_sl_atr_mult: float = 2.0  # SL = ATR * multiplier
    default_tp_atr_mult: float = 4.0  # TP = ATR * multiplier
    
    # === Spread / Slippage ===
    max_spread_points: float = 50.0  # Макс допустимый спред
    slippage_points: float = 5.0     # Учёт проскальзывания
    
    # === News Filters ===
    avoid_news_minutes_before: int = 15
    avoid_news_minutes_after: int = 15
    high_impact_news_symbols: List[str] = field(default_factory=lambda: ["USD", "XAU"])
    
    # === Backtesting ===
    use_tick_data: bool = True  # Использовать тики или M1 для интрабара
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    initial_balance: float = 10000.0
    
    # === Logging ===
    log_level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = True
    
    def validate(self) -> bool:
        """Валидация конфигурации"""
        assert 0 < self.risk_per_trade_pct <= 5.0, "risk_per_trade должен быть 0-5%"
        assert 0 < self.max_daily_loss_pct <= 10.0, "max_daily_loss должен быть 0-10%"
        assert self.max_positions > 0, "max_positions должен быть > 0"
        assert 0 <= self.min_direction_confidence <= 1.0, "confidence должен быть 0-1"
        assert self.session_start_utc < self.session_end_utc, "Некорректная торговая сессия"
        
        # Валидация мультитаймфреймовых параметров
        assert self.primary_tf in self.supported_timeframes, f"primary_tf={self.primary_tf} не в supported_timeframes"
        assert self.execution_tf in self.supported_timeframes, f"execution_tf={self.execution_tf} не в supported_timeframes"
        assert self.context_tf_high in self.supported_timeframes, f"context_tf_high={self.context_tf_high} не в supported_timeframes"
        assert 0 <= self.tf_selector_min_confidence <= 1.0, "tf_selector_min_confidence должен быть 0-1"
        assert 0 <= self.tf_selector_high_confidence <= 1.0, "tf_selector_high_confidence должен быть 0-1"
        
        return True
        return True


@dataclass
class RegimeSettings:
    """Настройки для конкретного режима рынка"""
    regime_name: str
    allowed_directions: List[str]  # ["long", "short", "both"]
    entry_style: str  # "breakout", "reversion", "momentum"
    max_positions: int
    risk_multiplier: float  # Корректировка риска для режима
    min_confidence: float

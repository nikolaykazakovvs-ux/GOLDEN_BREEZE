# strategy/__init__.py
"""
Golden Breeze Hybrid Strategy v1.0 + Multitimeframe

Гибридная торговая стратегия с интрабарной логикой и мультитаймфреймом:
- Динамический выбор PRIMARY_TF через TimeframeSelector
- Адаптивная под режимы рынка (trend/range/volatile)
- Работа внутри свечи (тики/M1)
- Интеграция с AI Core для мультитаймфреймовых сигналов
- Строгий risk management
"""

from .config import StrategyConfig
from .hybrid_strategy import HybridStrategy
from .risk_manager import RiskManager
from .timeframe_selector import TimeframeSelector, TimeframeData, Timeframe, Regime, TimeframeDecision
from .backtest_engine import BacktestEngine

__all__ = [
    'StrategyConfig', 
    'HybridStrategy', 
    'RiskManager',
    'TimeframeSelector',
    'TimeframeData',
    'Timeframe',
    'Regime',
    'TimeframeDecision',
    'BacktestEngine'
]

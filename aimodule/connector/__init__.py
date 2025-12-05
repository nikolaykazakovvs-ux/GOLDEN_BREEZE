# aimodule/connector/__init__.py

"""
AI Connector для Golden Breeze v3.0.

Предоставляет:
- Клиентский интерфейс для внешних систем (GoldenBreezeClient)
- Унифицированные коннекторы к брокерам и биржам (MT5, MEXC, TradeLocker)
- Базовые классы и типы для расширения
"""

from .client import GoldenBreezeClient

# Base classes
from .base import (
    BaseConnector,
    OrderSide,
    OrderType,
    OrderResult,
    Position,
    AccountInfo
)

# Connectors
from .mt5 import MT5Connector
from .mexc import MEXCConnector, MexcConnector
from .tradelocker import TradeLockerConnector, TradlockerConnector

__all__ = [
    # Client
    "GoldenBreezeClient",
    
    # Base
    "BaseConnector",
    "OrderSide",
    "OrderType",
    "OrderResult",
    "Position",
    "AccountInfo",
    
    # Connectors
    "MT5Connector",
    "MEXCConnector",
    "MexcConnector",
    "TradeLockerConnector",
    "TradlockerConnector",
]

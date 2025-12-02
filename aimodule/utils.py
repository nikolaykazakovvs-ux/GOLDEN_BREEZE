# aimodule/utils.py

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class MarketRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Action(str, Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    HOLD = "hold"
    SKIP = "skip"
    EXIT = "exit"


class Candle(BaseModel):
    timestamp: str = Field(..., description="ISO timestamp or broker format")
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Candle]


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    regime: MarketRegime
    direction: Direction
    sentiment: float
    confidence: float
    action: Action
    reasons: List[str] | None = Field(default=None, description="Explanation of why this action was chosen")


class FeedbackRequest(BaseModel):
    """
    Обратная связь по фактической сделке для self-learning слоя.
    """
    symbol: str = Field(..., description="Торгуемый инструмент, напр. XAUUSD")
    timeframe: Optional[str] = Field(None, description="Таймфрейм сделки, напр. M5")
    direction: Optional[Direction] = Field(
        None, description="Направление позиции, если применимо"
    )
    action: Optional[Action] = Field(
        None, description="Какое действие было принято (enter_long, exit и т.п.)"
    )
    pnl: float = Field(..., description="Результат сделки в валюте или пунктах")
    regime: Optional[MarketRegime] = Field(
        None, description="Режим рынка на момент входа"
    )
    sentiment: Optional[float] = Field(
        None, description="Sentiment score [-1.0, 1.0] на момент входа"
    )
    confidence: Optional[float] = Field(
        None, description="Уверенность AI-сигнала [0.0, 1.0] на момент входа"
    )
    entry_price: Optional[float] = Field(
        None, description="Цена входа (опционально)"
    )
    exit_price: Optional[float] = Field(
        None, description="Цена выхода (опционально)"
    )
    timestamp: str = Field(
        ..., description="Время исполнения сделки (ISO-строка или формат брокера)"
    )


class FeedbackResponse(BaseModel):
    status: str = Field(..., description="ok / error")
    message: Optional[str] = Field(None, description="Дополнительная информация")

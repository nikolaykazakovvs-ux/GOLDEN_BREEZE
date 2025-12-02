"""trading.market_data — OHLCV MCP with MT5 integration.

Provides OHLCV data from MetaTrader 5.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone
from .mt5_connector import get_connector

# Маппинг таймфреймов
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

def get_ohlcv(symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None, count: int = 1000) -> pd.DataFrame:
    """Получить OHLCV данные из MT5.
    
    Args:
        symbol: Символ инструмента (например, "XAUUSD")
        timeframe: Таймфрейм (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        start: Начальная дата ISO (например, "2024-01-01") — опционально
        end: Конечная дата ISO (например, "2024-12-31") — опционально
        count: Количество свечей (по умолчанию 1000), если не заданы start/end
    
    Returns:
        DataFrame с колонками [time, open, high, low, close, tick_volume, volume, spread]
    """
    # Инициализация MT5
    connector = get_connector()
    if not connector.is_connected():
        if not connector.initialize():
            print("Failed to connect to MT5")
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
    
    # Преобразование таймфрейма
    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        print(f"Unknown timeframe: {timeframe}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
    
    # Получение данных
    if start and end:
        # Диапазон дат
        date_from = datetime.fromisoformat(start.replace("Z", "+00:00"))
        date_to = datetime.fromisoformat(end.replace("Z", "+00:00"))
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    elif start:
        # От даты + count свечей
        date_from = datetime.fromisoformat(start.replace("Z", "+00:00"))
        rates = mt5.copy_rates_from(symbol, tf, date_from, count)
    else:
        # Последние count свечей
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol} {timeframe}, error: {mt5.last_error()}")
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
    
    # Преобразование в DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    
    # Переименование колонок для совместимости
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    
    return df[["time", "open", "high", "low", "close", "volume"]]

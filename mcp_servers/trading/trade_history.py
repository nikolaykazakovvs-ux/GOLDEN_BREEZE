"""trading.trade_history — Trade History MCP with MT5 integration.

Provides closed trades and open positions from MetaTrader 5.
"""
from __future__ import annotations
from typing import List, Dict, Optional
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from .mt5_connector import get_connector

def get_closed_trades(account_id: str, symbol: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict]:
    """Получить историю закрытых сделок из MT5.
    
    Args:
        account_id: ID аккаунта (не используется, берётся текущий подключённый)
        symbol: Фильтр по символу (опционально)
        start: Начальная дата ISO
        end: Конечная дата ISO
    
    Returns:
        Список словарей с информацией о сделках
    """
    # Инициализация MT5
    connector = get_connector()
    if not connector.is_connected():
        if not connector.initialize():
            print("Failed to connect to MT5")
            return []
    
    # Определение диапазона дат
    if start:
        date_from = datetime.fromisoformat(start.replace("Z", "+00:00"))
    else:
        date_from = datetime.now() - timedelta(days=30)  # По умолчанию последние 30 дней
    
    if end:
        date_to = datetime.fromisoformat(end.replace("Z", "+00:00"))
    else:
        date_to = datetime.now()
    
    # Получение истории сделок
    if symbol:
        deals = mt5.history_deals_get(date_from, date_to, group=symbol)
    else:
        deals = mt5.history_deals_get(date_from, date_to)
    
    if deals is None or len(deals) == 0:
        return []
    
    # Преобразование в список словарей
    result = []
    for deal in deals:
        result.append({
            "ticket": deal.ticket,
            "order": deal.order,
            "time": datetime.fromtimestamp(deal.time, tz=datetime.now().astimezone().tzinfo).isoformat(),
            "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
            "entry": "IN" if deal.entry == mt5.DEAL_ENTRY_IN else "OUT",
            "symbol": deal.symbol,
            "volume": deal.volume,
            "price": deal.price,
            "commission": deal.commission,
            "swap": deal.swap,
            "profit": deal.profit,
            "comment": deal.comment,
        })
    
    return result

def get_open_positions(account_id: str) -> List[Dict]:
    """Получить текущие открытые позиции из MT5.
    
    Args:
        account_id: ID аккаунта (не используется, берётся текущий подключённый)
    
    Returns:
        Список словарей с информацией о позициях
    """
    # Инициализация MT5
    connector = get_connector()
    if not connector.is_connected():
        if not connector.initialize():
            print("Failed to connect to MT5")
            return []
    
    # Получение открытых позиций
    positions = mt5.positions_get()
    
    if positions is None or len(positions) == 0:
        return []
    
    # Преобразование в список словарей
    result = []
    for pos in positions:
        result.append({
            "ticket": pos.ticket,
            "time": datetime.fromtimestamp(pos.time, tz=datetime.now().astimezone().tzinfo).isoformat(),
            "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            "symbol": pos.symbol,
            "volume": pos.volume,
            "price_open": pos.price_open,
            "price_current": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "commission": pos.commission,
            "swap": pos.swap,
            "profit": pos.profit,
            "comment": pos.comment,
        })
    
    return result

"""trading.metrics — Metrics / Monitoring MCP with full MT5 integration.

Provides comprehensive trading metrics:
- Date Start/End, ROI, Net PnL, Win Ratio
- Max Drawdown, Time in Market, Number of Trades
- Average Trade Duration, Timeframe
"""
from __future__ import annotations
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
from .mt5_connector import get_connector
from .trade_history import get_closed_trades

def calculate_metrics(trades: List[Dict], initial_balance: float = 10000.0, timeframe: str = "M5") -> Dict:
    """Рассчитать все метрики на основе истории сделок.
    
    Args:
        trades: Список сделок из get_closed_trades
        initial_balance: Начальный баланс (по умолчанию 10000)
        timeframe: Таймфрейм стратегии
    
    Returns:
        Dict с полным набором метрик
    """
    if not trades:
        return {
            "date_start": None,
            "date_end": None,
            "roi_percent": 0.0,
            "net_pnl": 0.0,
            "win_ratio_percent": 0.0,
            "max_drawdown_percent": 0.0,
            "time_in_market_percent": 0.0,
            "number_of_trades": 0,
            "average_trade_duration": "0h 0m",
            "timeframe": timeframe,
        }
    
    # Сортировка по времени
    trades_sorted = sorted(trades, key=lambda x: x['time'])
    
    # Date Start / End
    date_start = trades_sorted[0]['time']
    date_end = trades_sorted[-1]['time']
    
    # Net PnL
    net_pnl = sum(t['profit'] for t in trades_sorted)
    
    # ROI
    roi_percent = (net_pnl / initial_balance) * 100 if initial_balance > 0 else 0.0
    
    # Win Ratio
    winning_trades = [t for t in trades_sorted if t['profit'] > 0]
    win_ratio_percent = (len(winning_trades) / len(trades_sorted)) * 100
    
    # Number of Trades
    number_of_trades = len(trades_sorted)
    
    # Max Drawdown - рассчитываем кумулятивную equity и находим максимальную просадку
    equity_curve = []
    cumulative_pnl = initial_balance
    peak = initial_balance
    max_dd = 0.0
    
    for trade in trades_sorted:
        cumulative_pnl += trade['profit']
        equity_curve.append(cumulative_pnl)
        
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        
        dd = ((peak - cumulative_pnl) / peak) * 100 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    
    # Time in Market и Average Trade Duration
    # Для этого нужны сделки с entry IN/OUT
    entry_trades = [t for t in trades_sorted if t.get('entry') == 'IN']
    exit_trades = [t for t in trades_sorted if t.get('entry') == 'OUT']
    
    # Паруем входы и выходы по order ID
    paired_trades = []
    for entry in entry_trades:
        order_id = entry.get('order')
        matching_exit = next((e for e in exit_trades if e.get('order') == order_id), None)
        if matching_exit:
            entry_time = datetime.fromisoformat(entry['time'])
            exit_time = datetime.fromisoformat(matching_exit['time'])
            duration = exit_time - entry_time
            paired_trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'duration': duration
            })
    
    # Average Trade Duration
    if paired_trades:
        avg_duration = sum([pt['duration'].total_seconds() for pt in paired_trades]) / len(paired_trades)
        hours = int(avg_duration // 3600)
        minutes = int((avg_duration % 3600) // 60)
        average_trade_duration = f"{hours}h {minutes}m"
        
        # Time in Market
        total_time_in_trades = sum([pt['duration'].total_seconds() for pt in paired_trades])
        period_start = datetime.fromisoformat(date_start)
        period_end = datetime.fromisoformat(date_end)
        total_period = (period_end - period_start).total_seconds()
        time_in_market_percent = (total_time_in_trades / total_period) * 100 if total_period > 0 else 0.0
    else:
        average_trade_duration = "N/A"
        time_in_market_percent = 0.0
    
    return {
        "date_start": date_start,
        "date_end": date_end,
        "roi_percent": round(roi_percent, 2),
        "net_pnl": round(net_pnl, 2),
        "win_ratio_percent": round(win_ratio_percent, 2),
        "max_drawdown_percent": round(max_dd, 2),
        "time_in_market_percent": round(time_in_market_percent, 2),
        "number_of_trades": number_of_trades,
        "average_trade_duration": average_trade_duration,
        "timeframe": timeframe,
        "equity_curve": equity_curve,  # для построения графиков
    }

def get_equity_curve(account_id: str, start: Optional[str] = None, end: Optional[str] = None) -> List[float]:
    """Получить equity curve (кривую баланса) по истории сделок.
    
    Args:
        account_id: ID аккаунта
        start: Начальная дата ISO
        end: Конечная дата ISO
    
    Returns:
        Список значений equity после каждой сделки
    """
    trades = get_closed_trades(account_id, start=start, end=end)
    
    if not trades:
        return []
    
    # Получаем начальный баланс
    connector = get_connector()
    if connector.is_connected() or connector.initialize():
        info = connector.get_account_info()
        initial_balance = info.get('balance', 10000.0) if info else 10000.0
    else:
        initial_balance = 10000.0
    
    # Строим equity curve
    trades_sorted = sorted(trades, key=lambda x: x['time'])
    equity_curve = [initial_balance]
    cumulative = initial_balance
    
    for trade in trades_sorted:
        cumulative += trade['profit']
        equity_curve.append(cumulative)
    
    return equity_curve

def get_regime_stats(account_id: str, symbol: str, regime: str, start: Optional[str] = None, end: Optional[str] = None) -> Dict:
    """Получить статистику по конкретному режиму рынка.
    
    Args:
        account_id: ID аккаунта
        symbol: Символ (например, XAUUSD)
        regime: Режим рынка (например, "Trending", "Volatile", "Ranging")
        start: Начальная дата ISO
        end: Конечная дата ISO
    
    Returns:
        Dict с метриками для данного режима
    """
    # Получаем все сделки по символу
    trades = get_closed_trades(account_id, symbol=symbol, start=start, end=end)
    
    if not trades:
        return {
            "regime": regime,
            "symbol": symbol,
            "trades_count": 0,
            "win_ratio": 0.0,
            "net_pnl": 0.0,
            "avg_profit": 0.0,
        }
    
    # Фильтруем по режиму (если есть информация в комментариях сделок)
    # Пока возвращаем общую статистику
    winning = [t for t in trades if t['profit'] > 0]
    
    return {
        "regime": regime,
        "symbol": symbol,
        "trades_count": len(trades),
        "win_ratio": round((len(winning) / len(trades)) * 100, 2) if trades else 0.0,
        "net_pnl": round(sum(t['profit'] for t in trades), 2),
        "avg_profit": round(sum(t['profit'] for t in trades) / len(trades), 2) if trades else 0.0,
    }

def get_overall_metrics(account_id: str, start: Optional[str] = None, end: Optional[str] = None, timeframe: str = "M5") -> Dict:
    """Получить полный набор метрик по аккаунту.
    
    Args:
        account_id: ID аккаунта
        start: Начальная дата ISO (по умолчанию последние 30 дней)
        end: Конечная дата ISO (по умолчанию сейчас)
        timeframe: Таймфрейм стратегии
    
    Returns:
        Dict со всеми метриками
    """
    # По умолчанию последние 30 дней
    if not start:
        start = (datetime.now() - timedelta(days=30)).isoformat()
    
    if not end:
        end = datetime.now().isoformat()
    
    # Получаем сделки
    trades = get_closed_trades(account_id, start=start, end=end)
    
    # Получаем начальный баланс
    connector = get_connector()
    if connector.is_connected() or connector.initialize():
        info = connector.get_account_info()
        initial_balance = info.get('balance', 10000.0) if info else 10000.0
    else:
        initial_balance = 10000.0
    
    # Рассчитываем метрики
    metrics = calculate_metrics(trades, initial_balance, timeframe)
    
    # Добавляем информацию об аккаунте
    if connector.is_connected():
        info = connector.get_account_info()
        if info:
            metrics['account_info'] = {
                'login': info['login'],
                'server': info['server'],
                'current_balance': info['balance'],
                'current_equity': info['equity'],
                'currency': info['currency'],
            }
    
    return metrics

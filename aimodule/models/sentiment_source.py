# aimodule/models/sentiment_source.py
"""
Источник новостей и TTL-кэш для Sentiment Engine v3.

В реальной системе здесь будут RSS/HTTP клиенты.
Сейчас реализуем безопасный мок + TTL кэш, чтобы не блокировать инференс.
"""

from typing import List, Dict, Tuple
import time

# TTL кэш: ключ (symbol) -> (timestamp, [news])
_CACHE: Dict[str, Tuple[float, List[str]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 минут


def _now() -> float:
    return time.time()


def _is_fresh(ts: float) -> bool:
    return (_now() - ts) < _CACHE_TTL_SECONDS


def get_latest_news(symbol: str, limit: int = 5) -> List[str]:
    """
    Получение последних новостей для символа с TTL кэшем.
    Возвращает список строк (заголовков/кратких сообщений).
    
    В этом мок-реализации:
    - генерируем фиксированный набор "новостей".
    - используем TTL кэш, чтобы не перегружать источники.
    """
    # Кэш-хит
    if symbol in _CACHE:
        ts, news = _CACHE[symbol]
        if _is_fresh(ts):
            return news[:limit]

    # Мок-новости (в реальности: загружать из RSS/API)
    news = [
        f"{symbol}: Fed commentary signals cautious stance on rates",
        f"{symbol}: Risk sentiment mixed amid geopolitical tensions",
        f"{symbol}: Market participants eye upcoming macro releases",
        f"{symbol}: Volatility increases on uncertainty",
        f"{symbol}: Technical outlook points to consolidation",
        f"{symbol}: Analysts note resilience despite headwinds"
    ]
    # Обновляем кэш
    _CACHE[symbol] = (_now(), news)
    return news[:limit]


def clear_cache():
    """Очистка TTL-кэша новостей (для тестов)."""
    _CACHE.clear()

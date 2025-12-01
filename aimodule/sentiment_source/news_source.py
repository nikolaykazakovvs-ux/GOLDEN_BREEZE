# aimodule/sentiment_source/news_source.py

"""
Источник новостей для sentiment анализа.

В текущей реализации используются фиктивные новости (заглушки).
В будущем можно интегрировать:
- RSS feeds (Reuters, Bloomberg)
- News API (newsapi.org)
- Twitter/X API
- Специализированные финансовые API
"""

from typing import List
from datetime import datetime


# Фиктивные новости для разных инструментов (заглушки)
MOCK_NEWS = {
    "XAUUSD": [
        "Gold prices steady as investors await Fed decision on interest rates",
        "Central banks continue gold purchases amid economic uncertainty",
        "Dollar strength weighs on gold prices in Asian trading"
    ],
    "EURUSD": [
        "Euro gains against dollar on strong eurozone manufacturing data",
        "ECB signals potential rate cuts as inflation moderates",
        "Market volatility increases ahead of US jobs report"
    ],
    "BTCUSD": [
        "Bitcoin rallies past resistance as institutional interest grows",
        "Cryptocurrency market sees increased regulatory scrutiny",
        "Major exchange reports record trading volumes"
    ]
}


def get_latest_news(symbol: str, limit: int = 5) -> List[str]:
    """
    Получение последних новостей для инструмента.
    
    Args:
        symbol: торговый инструмент (XAUUSD, EURUSD и т.д.)
        limit: максимальное количество новостей
        
    Returns:
        список текстов новостей
        
    Note:
        Текущая реализация возвращает фиктивные новости.
        Для production замените на реальный источник данных.
    """
    # Нормализация символа
    symbol = symbol.upper().strip()
    
    # Получение новостей из mock данных
    news_list = MOCK_NEWS.get(symbol, [])
    
    # Если для символа нет специфичных новостей, возвращаем общие
    if not news_list:
        news_list = [
            "Market remains in consolidation phase with low volatility",
            "Technical indicators show neutral sentiment across major pairs"
        ]
    
    # Ограничение количества
    return news_list[:limit]


def get_news_with_timestamps(symbol: str, limit: int = 5) -> List[dict]:
    """
    Получение новостей с метаданными.
    
    Args:
        symbol: торговый инструмент
        limit: максимальное количество новостей
        
    Returns:
        список словарей с полями:
        - text: текст новости
        - timestamp: время публикации
        - source: источник
    """
    news_texts = get_latest_news(symbol, limit)
    
    # Добавление метаданных
    result = []
    for i, text in enumerate(news_texts):
        result.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "source": "mock_source",
            "relevance": 1.0 - (i * 0.1)  # убывающая релевантность
        })
    
    return result


# Пример интеграции с реальным API (закомментировано)
"""
def get_real_news_from_api(symbol: str, limit: int = 5) -> List[str]:
    '''
    Пример интеграции с реальным News API.
    
    Требуется:
    - pip install requests
    - API ключ от newsapi.org
    '''
    import requests
    
    API_KEY = "your_api_key_here"
    
    # Mapping символов на поисковые запросы
    search_queries = {
        "XAUUSD": "gold price OR gold market",
        "EURUSD": "euro dollar OR EUR/USD",
        "BTCUSD": "bitcoin OR cryptocurrency"
    }
    
    query = search_queries.get(symbol, symbol)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        
        # Извлечение заголовков и описаний
        news_texts = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title}. {description}".strip()
            if text:
                news_texts.append(text)
        
        return news_texts
    
    except Exception as e:
        print(f"Ошибка при получении новостей: {e}")
        return []
"""

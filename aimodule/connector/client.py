# aimodule/connector/client.py

"""
Клиент для подключения к Golden Breeze AI API.

Предоставляет простой интерфейс для внешних торговых систем.
"""

import requests
from typing import List, Dict, Any, Optional


class GoldenBreezeClient:
    """
    Клиент для взаимодействия с Golden Breeze AI API.
    
    Использование:
    ```python
    from aimodule.connector.client import GoldenBreezeClient
    
    client = GoldenBreezeClient()
    
    # Получение AI предсказания
    response = client.predict(
        symbol="XAUUSD",
        timeframe="M5",
        candles=[...]
    )
    
    # Отправка обратной связи
    client.send_feedback({
        "symbol": "XAUUSD",
        "direction": "long",
        "action": "enter_long",
        "pnl": 150.0,
        ...
    })
    ```
    
    ВАЖНО: Golden Breeze НЕ содержит торговой стратегии.
    Он предоставляет только AI-сигналы:
    - regime: режим рынка
    - direction: направление
    - sentiment: настроение
    - confidence: уверенность
    - action: рекомендация (но решение принимает внешняя система!)
    - reasons: объяснения
    
    Внешняя торговая система должна сама решать:
    - Управление позициями (position sizing)
    - Stop Loss / Take Profit
    - Risk management
    - Когда входить/выходить
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:5005", timeout: int = 10):
        """
        Args:
            base_url: URL AI сервера
            timeout: таймаут запросов в секундах
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def health(self) -> Dict[str, Any]:
        """
        Проверка здоровья AI сервера.
        
        Returns:
            {"status": "ok", "message": "..."}
            
        Raises:
            requests.RequestException: при ошибке соединения
        """
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Получение AI предсказания на основе исторических свечей.
        
        Args:
            symbol: торговый инструмент (XAUUSD, EURUSD и т.д.)
            timeframe: таймфрейм (M5, M15, H1 и т.д.)
            candles: список свечей в формате:
                [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "open": 2000.0,
                        "high": 2010.0,
                        "low": 1995.0,
                        "close": 2005.0,
                        "volume": 1000.0
                    },
                    ...
                ]
        
        Returns:
            {
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "regime": "trend_up",          # Режим рынка
                "direction": "long",           # Направление
                "sentiment": 0.3,              # Настроение [-1, 1]
                "confidence": 0.75,            # Уверенность [0, 1]
                "action": "enter_long",        # Рекомендация
                "reasons": [                   # Объяснения
                    "Strong uptrend detected",
                    "High confidence (75%)",
                    ...
                ]
            }
        
        Raises:
            requests.RequestException: при ошибке соединения
            ValueError: при невалидных данных
        """
        url = f"{self.base_url}/predict"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def send_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Отправка обратной связи о результате сделки.
        
        Используется для self-learning: AI анализирует результаты
        и корректирует свои пороги и параметры.
        
        Args:
            feedback: словарь с данными:
                {
                    "symbol": "XAUUSD",
                    "direction": "long",       # long/short/flat
                    "action": "enter_long",    # действие, которое было выполнено
                    "regime": "trend_up",      # режим на момент входа
                    "sentiment": 0.3,          # sentiment на момент входа
                    "confidence": 0.75,        # confidence на момент входа
                    "entry_price": 2050.0,     # цена входа (опционально)
                    "exit_price": 2060.0,      # цена выхода (опционально)
                    "pnl": 150.0,              # прибыль/убыток (опционально)
                    "timestamp": "2024-01-01T12:00:00"  # время (опционально)
                }
        
        Returns:
            {"status": "ok", "message": "Feedback recorded"}
        
        Raises:
            requests.RequestException: при ошибке соединения
        """
        url = f"{self.base_url}/feedback"
        
        response = requests.post(url, json=feedback, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации AI (включая динамические пороги).
        
        Returns:
            Словарь с конфигурацией (если endpoint реализован)
        
        Note:
            Этот endpoint опционален и может быть не реализован.
        """
        url = f"{self.base_url}/config"
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": "Config endpoint not implemented"}
            raise


# Удобные alias-функции для быстрого использования

def quick_predict(
    symbol: str,
    candles: List[Dict[str, Any]],
    timeframe: str = "M5",
    base_url: str = "http://127.0.0.1:5005"
) -> Dict[str, Any]:
    """
    Быстрое получение предсказания без создания экземпляра клиента.
    
    Args:
        symbol: торговый инструмент
        candles: список свечей
        timeframe: таймфрейм
        base_url: URL AI сервера
    
    Returns:
        AI предсказание
    """
    client = GoldenBreezeClient(base_url=base_url)
    return client.predict(symbol=symbol, timeframe=timeframe, candles=candles)


def quick_feedback(
    feedback: Dict[str, Any],
    base_url: str = "http://127.0.0.1:5005"
) -> Dict[str, Any]:
    """
    Быстрая отправка обратной связи без создания экземпляра клиента.
    
    Args:
        feedback: данные обратной связи
        base_url: URL AI сервера
    
    Returns:
        Ответ сервера
    """
    client = GoldenBreezeClient(base_url=base_url)
    return client.send_feedback(feedback)

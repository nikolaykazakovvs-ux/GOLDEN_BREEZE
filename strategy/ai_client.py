# strategy/ai_client.py
"""
Клиент для взаимодействия с Golden Breeze AI Core
"""

import requests
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class AIClient:
    """
    Клиент для получения сигналов от AI Core.
    Поддерживает мультитаймфреймовые запросы.
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:5005"):
        self.api_url = api_url.rstrip("/")
        self.last_signal: Optional[Dict] = None
        self.last_multitf_signals: Optional[Dict[str, Dict]] = None  # {tf: signal}
    
    def health_check(self) -> bool:
        """Проверка доступности AI сервера"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict(self, symbol: str, timeframe: str, candles: List[Dict]) -> Optional[Dict]:
        """
        Получение сигнала от AI для одного таймфрейма.
        
        Args:
            symbol: XAUUSD
            timeframe: M5, M15, H1, H4
            candles: Список свечей [{timestamp, open, high, low, close, volume}, ...]
            
        Returns:
            {
                "regime": "trend_up/trend_down/range/volatile",
                "direction": "long/short/flat",
                "direction_confidence": 0.0-1.0,
                "sentiment": -1.0 to 1.0,
                "action": "enter_long/enter_short/hold/skip",
                "reasons": ["..."]
            }
        """
        try:
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles
            }
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                signal = response.json()
                self.last_signal = signal
                return signal
            else:
                print(f"AI predict error: {response.status_code}")
                return None
        
        except Exception as e:
            print(f"AI client error: {e}")
            return None
    
    def predict_multitimeframe(
        self, 
        symbol: str, 
        timeframes_data: Dict[str, List[Dict]]
    ) -> Optional[Dict[str, Dict]]:
        """
        Получение сигналов от AI по нескольким таймфреймам одновременно.
        
        Args:
            symbol: XAUUSD
            timeframes_data: {
                "M5": [{timestamp, open, high, low, close, volume}, ...],
                "M15": [...],
                "H1": [...],
                "H4": [...]
            }
            
        Returns:
            {
                "M5": {regime, direction, direction_confidence, sentiment, ...},
                "M15": {...},
                "H1": {...},
                "H4": {...}
            }
        """
        try:
            payload = {
                "symbol": symbol,
                "timeframes_data": timeframes_data
            }
            
            response = requests.post(
                f"{self.api_url}/predict_multitimeframe",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                signals = response.json()
                self.last_multitf_signals = signals
                return signals
            else:
                print(f"AI predict_multitimeframe error: {response.status_code}")
                # Fallback: запрашиваем каждый TF отдельно
                return self._predict_multitf_fallback(symbol, timeframes_data)
        
        except Exception as e:
            print(f"AI multitimeframe client error: {e}")
            # Fallback: запрашиваем каждый TF отдельно
            return self._predict_multitf_fallback(symbol, timeframes_data)
    
    def _predict_multitf_fallback(
        self,
        symbol: str,
        timeframes_data: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        Fallback: запрашивает каждый таймфрейм отдельно.
        """
        results = {}
        for tf, candles in timeframes_data.items():
            signal = self.predict(symbol, tf, candles)
            if signal:
                results[tf] = signal
        return results if results else None
    
    def send_feedback(self, trade_data: Dict) -> bool:
        """
        Отправка feedback в AI для self-learning
        
        Args:
            trade_data: {
                "symbol": "XAUUSD",
                "regime": "trend_up",
                "direction": "long",
                "sentiment": 0.5,
                "result_pnl": 100.0,
                "good_trade": True/False
            }
        """
        try:
            response = requests.post(
                f"{self.api_url}/feedback",
                json=trade_data,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def get_last_signal(self) -> Optional[Dict]:
        """Последний полученный сигнал (single timeframe)"""
        return self.last_signal
    
    def get_last_multitf_signals(self) -> Optional[Dict[str, Dict]]:
        """Последние полученные сигналы по всем таймфреймам"""
        return self.last_multitf_signals

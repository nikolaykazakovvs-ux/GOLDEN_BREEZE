# test_ai_core.py

"""
Тесты для AI-ядра Golden Breeze v2.0

Запуск:
    pytest test_ai_core.py -v
    или
    python -m pytest test_ai_core.py -v
"""

import pytest
from fastapi.testclient import TestClient
from aimodule.server.local_ai_gateway import app
from aimodule.utils import MarketRegime, Direction, Action


# Тестовый клиент FastAPI
client = TestClient(app)


@pytest.fixture
def sample_candles_small():
    """Небольшой набор свечей для быстрого теста."""
    return [
        {
            "timestamp": "2024-01-01T00:00:00",
            "open": 2000.0,
            "high": 2010.0,
            "low": 1995.0,
            "close": 2005.0,
            "volume": 1000.0
        },
        {
            "timestamp": "2024-01-01T01:00:00",
            "open": 2005.0,
            "high": 2015.0,
            "low": 2000.0,
            "close": 2010.0,
            "volume": 1200.0
        },
        {
            "timestamp": "2024-01-01T02:00:00",
            "open": 2010.0,
            "high": 2020.0,
            "low": 2005.0,
            "close": 2015.0,
            "volume": 1100.0
        }
    ]


@pytest.fixture
def sample_candles_large():
    """Большой набор свечей для полноценного теста."""
    candles = []
    base_price = 2000.0
    
    for i in range(50):
        # Симуляция восходящего тренда
        price = base_price + i * 2
        candles.append({
            "timestamp": f"2024-01-01T{i:02d}:00:00",
            "open": price,
            "high": price + 5,
            "low": price - 3,
            "close": price + 2,
            "volume": 1000.0 + i * 10
        })
    
    return candles


@pytest.fixture
def sample_candles_volatile():
    """Набор волатильных свечей."""
    candles = []
    base_price = 2000.0
    
    for i in range(40):
        # Симуляция волатильного рынка
        direction = 1 if i % 2 == 0 else -1
        price = base_price + direction * (i % 10) * 5
        candles.append({
            "timestamp": f"2024-01-01T{i:02d}:00:00",
            "open": price,
            "high": price + 15,
            "low": price - 15,
            "close": price + direction * 3,
            "volume": 1000.0 + i * 20
        })
    
    return candles


class TestHealthEndpoint:
    """Тесты health endpoint."""
    
    def test_health_check(self):
        """Проверка работы health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "ok"
        assert "message" in data


class TestPredictEndpoint:
    """Тесты predict endpoint."""
    
    def test_predict_with_small_dataset(self, sample_candles_small):
        """Тест с небольшим набором данных."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_small
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверка наличия всех полей
        assert "symbol" in data
        assert "timeframe" in data
        assert "regime" in data
        assert "direction" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "action" in data
        assert "reasons" in data
        
        # Проверка значений
        assert data["symbol"] == "XAUUSD"
        assert data["timeframe"] == "M5"
    
    def test_predict_with_large_dataset(self, sample_candles_large):
        """Тест с большим набором данных (тренд)."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Проверка типов и диапазонов
        assert data["regime"] in [r.value for r in MarketRegime]
        assert data["direction"] in [d.value for d in Direction]
        assert data["action"] in [a.value for a in Action]
        
        assert -1.0 <= data["sentiment"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        
        # Проверка reasons
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0
    
    def test_predict_with_volatile_dataset(self, sample_candles_volatile):
        """Тест с волатильным рынком."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_volatile
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # При волатильности возможны SKIP или HOLD действия
        assert data["action"] in [a.value for a in Action]
        
        # Проверка reasons содержат объяснение
        assert isinstance(data["reasons"], list)
        reasons_text = " ".join(data["reasons"])
        assert len(reasons_text) > 0


class TestResponseValidation:
    """Тесты валидации ответов."""
    
    def test_regime_values(self, sample_candles_large):
        """Проверка допустимых значений regime."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        valid_regimes = {"trend_up", "trend_down", "range", "volatile", "unknown"}
        assert data["regime"] in valid_regimes
    
    def test_direction_values(self, sample_candles_large):
        """Проверка допустимых значений direction."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        valid_directions = {"long", "short", "flat"}
        assert data["direction"] in valid_directions
    
    def test_sentiment_range(self, sample_candles_large):
        """Проверка диапазона sentiment."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        assert -1.0 <= data["sentiment"] <= 1.0
    
    def test_confidence_range(self, sample_candles_large):
        """Проверка диапазона confidence."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_action_values(self, sample_candles_large):
        """Проверка допустимых значений action."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        valid_actions = {"enter_long", "enter_short", "hold", "skip", "exit"}
        assert data["action"] in valid_actions
    
    def test_reasons_not_empty(self, sample_candles_large):
        """Проверка что reasons не пустые."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        data = response.json()
        
        assert "reasons" in data
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0
        
        # Каждая причина должна быть строкой
        for reason in data["reasons"]:
            assert isinstance(reason, str)
            assert len(reason) > 0


class TestErrorHandling:
    """Тесты обработки ошибок."""
    
    def test_missing_candles(self):
        """Тест с отсутствующими свечами."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": []
        }
        
        response = client.post("/predict", json=payload)
        
        # Может быть 422 (validation error) или 500 (internal error)
        assert response.status_code in [422, 500]
    
    def test_invalid_candle_data(self):
        """Тест с невалидными данными свечей."""
        payload = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "candles": [
                {
                    "timestamp": "invalid",
                    "open": "not_a_number",
                    "high": 2010.0,
                    "low": 1995.0,
                    "close": 2005.0,
                    "volume": 1000.0
                }
            ]
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 422  # Validation error


class TestDifferentSymbols:
    """Тесты с разными торговыми инструментами."""
    
    def test_eurusd(self, sample_candles_large):
        """Тест с EURUSD."""
        payload = {
            "symbol": "EURUSD",
            "timeframe": "M15",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
    
    def test_btcusd(self, sample_candles_large):
        """Тест с BTCUSD."""
        payload = {
            "symbol": "BTCUSD",
            "timeframe": "H1",
            "candles": sample_candles_large
        }
        
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTCUSD"


if __name__ == "__main__":
    # Запуск тестов программно
    pytest.main([__file__, "-v", "--tb=short"])

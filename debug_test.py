import requests
import json

body = {
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "candles": [
        {"timestamp": "2025-11-30T10:00:00", "open": 2650.5, "high": 2652.0, "low": 2649.0, "close": 2651.5, "volume": 1000.0},
        {"timestamp": "2025-11-30T10:05:00", "open": 2651.5, "high": 2653.0, "low": 2651.0, "close": 2652.5, "volume": 1100.0}
    ]
}

try:
    r = requests.post('http://127.0.0.1:5005/predict', json=body)
    print(f"Status Code: {r.status_code}")
    print(f"Response:\n{r.text}")
except Exception as e:
    print(f"Exception: {e}")

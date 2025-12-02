# aimodule/server/local_ai_gateway.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict, List
import traceback
from pathlib import Path
import json

from ..utils import (
    PredictionRequest,
    PredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    Candle,
)
from ..data_pipeline.loader import candles_to_dataframe
from ..inference.predict_regime import infer_regime
from ..inference.predict_direction import infer_direction
from ..models.sentiment_engine import SentimentEngine
from ..inference.combine_signals import decide_action

# 🔹 импорт self-learning слоя
from ..learning.feedback_store import FeedbackStore
from ..learning.online_update import OnlineUpdater

# Инициализация sentiment engine (временно без HF для быстрого теста GPU)
sentiment_engine = SentimentEngine(use_hf_model=False)
# Инициализация self-learning компонентов
feedback_store = FeedbackStore()
online_updater = OnlineUpdater(feedback_store=feedback_store)

from ..config import DEVICE, USE_GPU

app = FastAPI(title="AICore_XAUUSD_v3.0", version="3.0.0")


@app.get("/health")
def health() -> Dict:
    """Health check with device info and model metadata."""
    response = {
        "status": "ok",
        "message": "AICore v3.0 is running with self-learning",
        "device": str(DEVICE),
        "use_gpu": bool(USE_GPU),
    }
    
    # Add direction model metadata if available
    from ..config import MODELS_DIR
    direction_meta_path = MODELS_DIR / "direction_lstm_hybrid_XAUUSD.json"
    if direction_meta_path.exists():
        try:
            with open(direction_meta_path, 'r') as f:
                meta = json.load(f)
                response["direction_model"] = {
                    "type": meta.get("model_type", "unknown"),
                    "training_date": meta.get("training_date", "unknown"),
                    "test_accuracy": meta.get("test_metrics", {}).get("accuracy"),
                    "test_mcc": meta.get("test_metrics", {}).get("mcc"),
                    "seq_len": meta.get("seq_len"),
                }
        except Exception:
            pass
    
    return response


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        # Валидация: список свечей не должен быть пустым
        if not req.candles:
            return JSONResponse(status_code=422, content={"error": "candles list must not be empty"})

        df = candles_to_dataframe(req.candles)

        # Определение режима рынка
        regime = infer_regime(df)
        
        # Определение направления
        direction, confidence = infer_direction(df)
        
        # Анализ sentiment через unified engine
        sentiment = sentiment_engine.get_sentiment(req.symbol, regime, context_text=None)
        
        # Принятие решения с учётом всех факторов (теперь возвращает reasons)
        action, reasons = decide_action(regime, direction, sentiment, confidence)

        resp = PredictionResponse(
            symbol=req.symbol,
            timeframe=req.timeframe,
            regime=regime,
            direction=direction,
            sentiment=sentiment,
            confidence=confidence,
            action=action,
            reasons=reasons,
        )
        return JSONResponse(content=resp.model_dump())
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """
    Приём обратной связи о сделке для self-learning:
    - сохраняем фидбек (FeedbackStore)
    - запускаем лёгкий online-update (OnlineUpdater)
    """
    try:
        # Добавление feedback в хранилище
        feedback_store.add_feedback(
            symbol=req.symbol,
            direction=req.direction.value if req.direction else "unknown",
            action=req.action.value if req.action else "unknown",
            regime=req.regime.value if req.regime else "unknown",
            sentiment=req.sentiment if req.sentiment is not None else 0.0,
            confidence=req.confidence if req.confidence is not None else 0.0,
            entry_price=req.entry_price,
            exit_price=req.exit_price,
            pnl=req.pnl,
            timestamp=req.timestamp
        )
        
        # Лёгкий online-update: обновление статистик, порогов и т.п.
        update_result = online_updater.update_thresholds()
        
        message = "Feedback accepted for self-learning"
        if update_result.get("updated"):
            message += f" | Thresholds updated: {len(update_result.get('updates', {}))} changes"
        
        return FeedbackResponse(status="ok", message=message)
    
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "traceback": tb}
        )


@app.post("/predict_multitimeframe")
def predict_multitimeframe(req: Dict):
    """
    Batch prediction for multiple timeframes.
    
    Request format:
    {
      "symbol": "XAUUSD",
      "timeframes_data": {
        "M5": [{"timestamp": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...],
        "M15": [...],
        "H1": [...],
        "H4": [...]
      }
    }
    
    Response format:
    {
      "M5": {PredictionResponse},
      "M15": {PredictionResponse},
      "H1": {PredictionResponse},
      "H4": {PredictionResponse}
    }
    """
    try:
        symbol = req.get("symbol", "XAUUSD")
        timeframes_data = req.get("timeframes_data", {})
        
        if not timeframes_data:
            return JSONResponse(
                status_code=422,
                content={"error": "timeframes_data must not be empty"}
            )
        
        results = {}
        
        # Process each timeframe
        for timeframe, candles_data in timeframes_data.items():
            if not candles_data:
                results[timeframe] = {"error": "empty candles list"}
                continue
            
            try:
                # Convert dict candles to Candle objects
                candles = [Candle(**c) for c in candles_data]
                
                df = candles_to_dataframe(candles)
                
                # Determine regime
                regime = infer_regime(df)
                
                # Determine direction
                direction, confidence = infer_direction(df)
                
                # Analyze sentiment
                sentiment = sentiment_engine.get_sentiment(symbol, regime, context_text=None)
                
                # Make decision
                action, reasons = decide_action(regime, direction, sentiment, confidence)
                
                resp = PredictionResponse(
                    symbol=symbol,
                    timeframe=timeframe,
                    regime=regime,
                    direction=direction,
                    sentiment=sentiment,
                    confidence=confidence,
                    action=action,
                    reasons=reasons,
                )
                
                results[timeframe] = resp.model_dump()
            
            except Exception as e:
                results[timeframe] = {"error": str(e)}
        
        return JSONResponse(content=results)
    
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": tb}
        )


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🚀 Starting AI Core Gateway v3.0")
    print("="*60)
    print("📍 URL: http://127.0.0.1:5005")
    print("📋 Health check: http://127.0.0.1:5005/health")
    print("🔍 Docs: http://127.0.0.1:5005/docs")
    print("="*60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5005,
        log_level="info"
    )

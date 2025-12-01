# aimodule/server/local_ai_gateway.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Dict
import traceback

from ..utils import (
    PredictionRequest,
    PredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from ..data_pipeline.loader import candles_to_dataframe
from ..inference.predict_regime import infer_regime
from ..inference.predict_direction import infer_direction
from ..models.sentiment_engine import SentimentEngine
from ..inference.combine_signals import decide_action

# 🔹 импорт self-learning слоя
from ..learning.feedback_store import FeedbackStore
from ..learning.online_update import OnlineUpdater

# Инициализация sentiment engine
sentiment_engine = SentimentEngine(use_hf_model=True)
# Инициализация self-learning компонентов
feedback_store = FeedbackStore()
online_updater = OnlineUpdater(feedback_store=feedback_store)

app = FastAPI(title="AICore_XAUUSD_v3.0", version="3.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "AICore v3.0 is running with self-learning"}


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

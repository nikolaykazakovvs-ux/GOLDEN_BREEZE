# aimodule/models/sentiment_engine.py

"""
Комбинированный sentiment движок.
Использует HF модель при наличии новостей, иначе fallback на лексикон или режим.
"""

from typing import Optional
from ..utils import MarketRegime
from .sentiment_model import LexiconSentimentModel
from .sentiment_hf_model import HFLocalSentimentModel, TRANSFORMERS_AVAILABLE
from .sentiment_source import get_latest_news


class SentimentEngine:
    """
    Unified sentiment движок с многоуровневой стратегией:
    
    1. Приоритет: HF модель + реальные новости
    2. Fallback 1: Lexicon модель (если есть обученный лексикон)
    3. Fallback 2: Режим-зависимый baseline
    """
    
    def __init__(
        self,
        use_hf_model: bool = True,
        hf_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        """
        Args:
            use_hf_model: использовать ли HuggingFace модель
            hf_model_name: имя HF модели
        """
        self.use_hf_model = use_hf_model and TRANSFORMERS_AVAILABLE
        
        # Инициализация моделей
        self.lexicon_model = LexiconSentimentModel()
        self.hf_model = None
        
        if self.use_hf_model:
            try:
                self.hf_model = HFLocalSentimentModel(model_name=hf_model_name)
            except Exception as e:
                print(f"⚠️  HF sentiment модель не загружена: {e}")
                self.use_hf_model = False
    
    def get_sentiment(
        self,
        symbol: str,
        regime: MarketRegime,
        context_text: Optional[str] = None
    ) -> float:
        """
        Получение sentiment score для инструмента.
        
        Args:
            symbol: торговый инструмент (XAUUSD и т.д.)
            regime: текущий режим рынка
            context_text: опциональный текст для анализа (если уже есть)
            
        Returns:
            sentiment в диапазоне [-1.0, 1.0]
        """
        # Стратегия 1: HF модель + новости
        if self.use_hf_model and self.hf_model is not None:
            try:
                # Если текст не предоставлен, получаем новости
                if context_text is None:
                    news = get_latest_news(symbol, limit=5)
                    if news:
                        context_text = " ".join(news)
                
                # Анализ через HF модель
                if context_text and len(context_text.strip()) > 10:
                    sentiment = self.hf_model.predict(context_text)
                    return sentiment
            
            except Exception as e:
                print(f"⚠️  Ошибка HF sentiment: {e}")
        
        # Стратегия 2: Lexicon модель
        try:
            # Получаем новости если нет текста
            if context_text is None:
                news = get_latest_news(symbol, limit=3)
                if news:
                    context_text = " ".join(news)
            
            # Используем лексикон
            if context_text:
                sentiment = self.lexicon_model.predict(regime, context_text)
                if sentiment != 0.0 or self.lexicon_model.lexicon:
                    return sentiment
        
        except Exception:
            pass
        
        # Стратегия 3: Baseline по режиму (fallback)
        return self._regime_based_sentiment(regime)
    
    def _regime_based_sentiment(self, regime: MarketRegime) -> float:
        """
        Базовый sentiment на основе режима рынка.
        
        Args:
            regime: текущий режим
            
        Returns:
            sentiment baseline
        """
        if regime == MarketRegime.TREND_UP:
            return 0.2
        if regime == MarketRegime.TREND_DOWN:
            return -0.2
        if regime == MarketRegime.VOLATILE:
            return -0.1
        return 0.0
    
    def get_sentiment_detailed(
        self,
        symbol: str,
        regime: MarketRegime
    ) -> dict:
        """
        Подробная информация о sentiment с указанием источника.
        
        Returns:
            dict с полями:
            - sentiment: float
            - source: str (hf_model, lexicon, regime_baseline)
            - confidence: float
        """
        # Попытка HF модель
        if self.use_hf_model and self.hf_model is not None:
            try:
                news = get_latest_news(symbol, limit=5)
                if news:
                    text = " ".join(news)
                    sentiment = self.hf_model.predict(text)
                    return {
                        "sentiment": sentiment,
                        "source": "hf_model",
                        "confidence": 0.8,
                        "news_count": len(news)
                    }
            except Exception:
                pass
        
        # Попытка Lexicon
        try:
            news = get_latest_news(symbol, limit=3)
            if news:
                text = " ".join(news)
                sentiment = self.lexicon_model.predict(regime, text)
                if self.lexicon_model.lexicon:
                    return {
                        "sentiment": sentiment,
                        "source": "lexicon",
                        "confidence": 0.5,
                        "news_count": len(news)
                    }
        except Exception:
            pass
        
        # Fallback
        sentiment = self._regime_based_sentiment(regime)
        return {
            "sentiment": sentiment,
            "source": "regime_baseline",
            "confidence": 0.3,
            "news_count": 0
        }

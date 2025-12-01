# aimodule/models/sentiment_hf_model.py

"""
Локальная sentiment модель на основе HuggingFace transformers.
Использует предобученную модель для анализа финансовых новостей.
"""

from typing import Optional
from pathlib import Path
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers не установлен, sentiment HF модель недоступна")


class HFLocalSentimentModel:
    """
    Локальная sentiment модель на базе HuggingFace.
    
    Использует предобученную модель:
    - cardiffnlp/twitter-roberta-base-sentiment-latest (общий sentiment)
    - или ProsusAI/finbert (финансовый sentiment)
    
    При первом запуске скачивает модель в ~/.cache/huggingface
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ):
        """
        Args:
            model_name: имя модели на HuggingFace Hub
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  HFLocalSentimentModel недоступна: transformers не установлен")
            return
        
        try:
            self._load_model()
        except Exception as e:
            print(f"⚠️  Не удалось загрузить sentiment модель: {e}")
            print("   Для установки выполните: pip install transformers torch")
    
    def _load_model(self):
        """Загрузка модели и токенизатора."""
        print(f"📥 Загрузка sentiment модели: {self.model_name}")
        print("   (при первом запуске модель будет скачана)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Sentiment модель загружена")
    
    def predict(self, text: str) -> float:
        """
        Анализ sentiment текста.
        
        Args:
            text: текст для анализа (новость, комментарий и т.д.)
            
        Returns:
            float в диапазоне [-1.0, 1.0]
            - -1.0: очень негативный
            - 0.0: нейтральный
            - 1.0: очень позитивный
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return 0.0
        
        if not text or len(text.strip()) < 3:
            return 0.0
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Инференс
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Преобразование в вероятности
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            
            # Интерпретация результата
            # Формат зависит от модели:
            # - twitter-roberta: [negative, neutral, positive]
            # - finbert: [positive, negative, neutral]
            
            if "twitter-roberta" in self.model_name.lower():
                # [negative, neutral, positive]
                negative, neutral, positive = probs[0], probs[1], probs[2]
            elif "finbert" in self.model_name.lower():
                # [positive, negative, neutral]
                positive, negative, neutral = probs[0], probs[1], probs[2]
            else:
                # По умолчанию предполагаем [negative, neutral, positive]
                negative, neutral, positive = probs[0], probs[1] if len(probs) > 1 else 0, probs[2] if len(probs) > 2 else 0
            
            # Расчёт sentiment score в [-1, 1]
            sentiment = positive - negative
            
            return float(np.clip(sentiment, -1.0, 1.0))
        
        except Exception as e:
            print(f"⚠️  Ошибка при анализе sentiment: {e}")
            return 0.0
    
    def predict_batch(self, texts: list[str]) -> list[float]:
        """
        Анализ sentiment для списка текстов (батч-инференс).
        
        Args:
            texts: список текстов
            
        Returns:
            список sentiment scores в [-1.0, 1.0]
        """
        if not texts:
            return []
        
        # Для простоты используем последовательную обработку
        # Можно оптимизировать через батч-токенизацию
        return [self.predict(text) for text in texts]
    
    def predict_average(self, texts: list[str]) -> float:
        """
        Средний sentiment для списка текстов.
        
        Args:
            texts: список текстов (например, новости за день)
            
        Returns:
            средний sentiment в [-1.0, 1.0]
        """
        if not texts:
            return 0.0
        
        scores = self.predict_batch(texts)
        valid_scores = [s for s in scores if s != 0.0]
        
        if not valid_scores:
            return 0.0
        
        return float(np.mean(valid_scores))


# Singleton instance (опционально)
_hf_sentiment_instance = None


def get_hf_sentiment_model(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
) -> HFLocalSentimentModel:
    """
    Получение singleton instance HF sentiment модели.
    
    Args:
        model_name: имя модели (используется только при первом вызове)
        
    Returns:
        HFLocalSentimentModel instance
    """
    global _hf_sentiment_instance
    
    if _hf_sentiment_instance is None:
        _hf_sentiment_instance = HFLocalSentimentModel(model_name=model_name)
    
    return _hf_sentiment_instance

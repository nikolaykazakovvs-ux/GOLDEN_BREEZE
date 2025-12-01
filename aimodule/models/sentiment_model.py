# aimodule/models/sentiment_model.py

from pathlib import Path
from typing import Optional

from ..utils import MarketRegime
from ..config import SENTIMENT_MODEL_PATH


class LexiconSentimentModel:
    """
    Локальная модель сентимента без внешних API.
    Использует простой лексикон (словарь слов с весами), сохранённый в файле.
    Если файл отсутствует, применяет режим-зависимый baseline.
    """

    def __init__(self):
        self.lexicon = {}
        path = Path(SENTIMENT_MODEL_PATH)
        if path.exists():
            try:
                # Ожидается формат: строка "word,weight" на каждой строке
                for line in path.read_text(encoding="utf-8").splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        word, weight = parts[0], float(parts[1])
                        self.lexicon[word.lower()] = weight
            except Exception:
                self.lexicon = {}

    def score_text(self, text: str) -> Optional[float]:
        if not self.lexicon:
            return None
        total = 0.0
        count = 0
        for token in text.lower().split():
            if token in self.lexicon:
                total += self.lexicon[token]
                count += 1
        if count == 0:
            return 0.0
        # Нормализуем и ограничиваем диапазон
        return max(-1.0, min(1.0, total / max(count, 1)))

    def predict(self, regime: MarketRegime, context_text: Optional[str] = None) -> float:
        # Если есть текст и лексикон — используем его
        if context_text is not None:
            s = self.score_text(context_text)
            if s is not None:
                return float(s)

        # Иначе — baseline по режиму
        if regime == MarketRegime.TREND_UP:
            return 0.2
        if regime == MarketRegime.TREND_DOWN:
            return -0.2
        if regime == MarketRegime.VOLATILE:
            return -0.1
        return 0.0

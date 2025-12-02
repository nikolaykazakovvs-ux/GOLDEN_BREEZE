# aimodule/training/build_sentiment_lexicon.py

"""
Создание простого лексикона для локального анализа сентимента.
Сохраняет файл models/sentiment_model.gguf (текстовый формат словарь: word,weight)
"""

from pathlib import Path
from ..config import SENTIMENT_MODEL_PATH, MODELS_DIR

DEFAULT_LEXICON = {
    # Позитив
    "bullish": 0.6,
    "rally": 0.5,
    "growth": 0.4,
    "up": 0.3,
    "breakout": 0.5,
    "strong": 0.4,
    "optimism": 0.5,
    # Негатив
    "bearish": -0.6,
    "drop": -0.5,
    "fall": -0.4,
    "down": -0.3,
    "breakdown": -0.5,
    "weak": -0.4,
    "fear": -0.5,
}


def save_lexicon(lexicon: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"{w},{v}" for w, v in lexicon.items()]
    Path(SENTIMENT_MODEL_PATH).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved lexicon to {SENTIMENT_MODEL_PATH}")


if __name__ == "__main__":
    save_lexicon(DEFAULT_LEXICON)

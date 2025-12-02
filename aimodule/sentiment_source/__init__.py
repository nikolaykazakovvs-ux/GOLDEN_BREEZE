# aimodule/sentiment_source/__init__.py

"""
Модуль источников данных для sentiment анализа.
"""

from .news_source import get_latest_news

__all__ = ["get_latest_news"]

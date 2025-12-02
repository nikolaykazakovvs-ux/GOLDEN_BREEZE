"""trading.news — News / Sentiment Source MCP stub.

Returns list of headlines/texts.
"""
from __future__ import annotations
from typing import List, Optional

def get_news(symbol: str, limit: int = 10, since: Optional[str] = None) -> List[str]:
    """Return list of news headlines (stub)."""
    return []

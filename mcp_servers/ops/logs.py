"""ops.logs — Logs / Observability MCP stub."""
from __future__ import annotations
from typing import List
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"

def get_logs(source: str = "system.log", since: str | None = None, level: str | None = None) -> List[str]:
    p = LOGS_DIR / source
    if not p.exists():
        return []
    try:
        return p.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

def get_recent_logs(source: str = "system.log", lines: int = 100) -> List[str]:
    """Get recent log lines from a log file."""
    p = LOGS_DIR / source
    if not p.exists():
        return []
    try:
        all_lines = p.read_text(encoding="utf-8").splitlines()
        return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except Exception:
        return []

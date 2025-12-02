"""ops.config — Config Management MCP stub."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]

def get_config(name: str) -> Dict:
    p = REPO_ROOT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def update_config(name: str, patch: Dict) -> Dict:
    current = get_config(name)
    current.update(patch)
    p = REPO_ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current

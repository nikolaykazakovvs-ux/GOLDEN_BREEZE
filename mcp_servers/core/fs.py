"""core.fs — Project File System MCP stub.

Provides safe access to repository files.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[3]

ALLOWED_WRITE_DIRS = {
    REPO_ROOT / "aimodule",
    REPO_ROOT / "tests",
    REPO_ROOT / "docs",
}

def _resolve(path: str | Path) -> Path:
    p = (REPO_ROOT / path) if isinstance(path, str) else path
    rp = p.resolve()
    if REPO_ROOT not in rp.parents and rp != REPO_ROOT:
        raise PermissionError("Access outside repository root is forbidden")
    return rp

def list(path: str = ".") -> List[str]:
    p = _resolve(path)
    return sorted([c.name for c in p.iterdir()])

def read_file(path: str) -> str:
    p = _resolve(path)
    return p.read_text(encoding="utf-8")

def exists(path: str) -> bool:
    try:
        _resolve(path)
        return True
    except Exception:
        return False

def write_file(path: str, content: str) -> None:
    p = _resolve(path)
    # allow only under allowed dirs
    if not any(str(p).startswith(str(d)) for d in ALLOWED_WRITE_DIRS):
        raise PermissionError("Writing to this path is not allowed by policy")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

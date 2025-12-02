"""core.git — Git MCP stub.

Provides minimal wrappers around git commands.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

def _run_git(args: list[str]) -> tuple[str, str, int]:
    proc = subprocess.Popen([
        "git", *args
    ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return out, err, proc.returncode

def git_status() -> str:
    out, err, code = _run_git(["status", "--porcelain", "-b"])
    if code != 0:
        raise RuntimeError(err)
    return out

def git_diff(files: Optional[list[str]] = None) -> str:
    args = ["diff"] + (files or [])
    out, err, code = _run_git(args)
    if code != 0:
        raise RuntimeError(err)
    return out

def git_commit(message: str) -> str:
    out, err, code = _run_git(["commit", "-m", message])
    if code != 0:
        raise RuntimeError(err)
    return out

def git_push(branch: str = "main") -> str:
    out, err, code = _run_git(["push", "origin", branch])
    if code != 0:
        raise RuntimeError(err)
    return out

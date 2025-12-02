"""core.shell — Shell / Process Runner MCP stub.

Runs commands in repository sandbox.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN = {"rm", "del", "rmdir", "mkfs", "format"}

def run(command: str, cwd: str | None = None) -> Dict[str, str | int]:
    # naive safety check
    head = command.strip().split()[0].lower()
    if head in FORBIDDEN:
        raise PermissionError("Forbidden command")
    workdir = Path(cwd).resolve() if cwd else REPO_ROOT
    if REPO_ROOT not in workdir.parents and workdir != REPO_ROOT:
        raise PermissionError("cwd must be inside repository")
    proc = subprocess.Popen(command, cwd=str(workdir), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return {"stdout": out, "stderr": err, "exit_code": proc.returncode}

"""core.python — Python Runtime MCP stub.

Executes small Python snippets for sanity checks.
"""
from __future__ import annotations
from typing import Dict

def python_exec(code: str) -> Dict[str, str]:
    import io, contextlib
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = ""
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ns = {}
            exec(code, ns, ns)
        result = ns.get("result", "")
        return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "result": str(result)}
    except Exception as e:
        return {"stdout": stdout.getvalue(), "stderr": f"{e}", "result": ""}

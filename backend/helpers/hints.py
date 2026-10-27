"""Allowed-value hints recovered from the mock tools' source code.

Several tools dispatch on a free-form string (e.g. github_issue_write's
`method` must be "create" or "update"), but the generated schema only says
"str". We read the tool function's source (read-only) and collect the string
literals each parameter is compared against, then surface them in the
parameter description so the model doesn't guess "POST".
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from functools import lru_cache


def _literals(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


@lru_cache(maxsize=256)
def allowed_values(tool_name: str) -> dict[str, list[str]]:
    from backend.main import get_tool_spec

    func = get_tool_spec(tool_name).callable
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return {}
    params = set(inspect.signature(func).parameters)
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id not in params:
            continue
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.In, ast.NotIn)):
                for value in _literals(comp):
                    bucket = found.setdefault(node.left.id, [])
                    if value not in bucket:
                        bucket.append(value)
    return {k: v for k, v in found.items() if 1 < len(v) <= 25 or k == "method"}

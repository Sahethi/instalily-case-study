"""Runs registered tools and records every invocation for the response log."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from backend.chat_schema import ToolCallLog
from backend.helpers.config import MAX_TOOL_OUTPUT_CHARS

log = logging.getLogger("chat.tools")

_READ_PREFIXES = ("list", "get", "fetch", "find", "search", "read", "download")


def is_read_only(name: str) -> bool:
    lowered = name.lower()
    for prefix in ("googlecalendar_", "googledrive_", "gmail_", "github_", "linear_", "slack_"):
        lowered = lowered.removeprefix(prefix)
    return lowered.startswith(_READ_PREFIXES) or lowered.endswith(("_read", "_list", "_get", "_history"))


def _clean_args(args: dict[str, Any]) -> dict[str, Any]:
    """Strict mode makes the model send null for every unused optional.
    Dropping them lets the tool's own defaults apply."""
    return {k: v for k, v in args.items() if v is not None}


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]]
        return "Invalid arguments - " + "; ".join(parts)
    return f"{type(exc).__name__}: {exc}"


class ToolExecutor:
    """Executes tools for one /chat turn and keeps the ordered call log."""

    def __init__(self) -> None:
        self.log: list[ToolCallLog] = []
        self._read_cache: dict[str, str] = {}

    def run(self, name: str, raw_args: dict[str, Any]) -> str:
        """Invoke a tool and return the text the model should see."""
        from backend.main import get_tool_spec

        args = _clean_args(raw_args)
        cache_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if is_read_only(name) and cache_key in self._read_cache:
            # Same read twice in one turn: answer from memory instead of re-calling.
            return self._read_cache[cache_key]

        started = time.perf_counter()
        try:
            spec = get_tool_spec(name)
        except KeyError:
            return json.dumps({"error": f"Unknown tool '{name}'. Use search_tools to find the right one."})

        try:
            result = spec.invoke(**args).model_dump(mode="json")["result"]
        except Exception as exc:
            error = _short_error(exc)
            self.log.append(ToolCallLog(name=name, arguments=args, result=None, error=error))
            log.info("tool %s failed in %.0fms: %s", name, (time.perf_counter() - started) * 1000, error)
            return json.dumps({"error": error})

        self.log.append(ToolCallLog(name=name, arguments=args, result=result, error=None))
        log.info("tool %s ok in %.0fms", name, (time.perf_counter() - started) * 1000)
        text = json.dumps(result, default=str)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + f"... [truncated, {len(text)} chars total]"
        if is_read_only(name):
            self._read_cache[cache_key] = text
        else:
            # A write may change what reads return, so forget earlier reads.
            self._read_cache.clear()
        return text

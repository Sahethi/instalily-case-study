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


# Reads whose names don't follow the list/get/fetch pattern.
_EXTRA_READS = {
    "GOOGLECALENDAR_FREE_BUSY_QUERY", "GOOGLECALENDAR_EVENTS_INSTANCES", "GOOGLECALENDAR_SYNC_EVENTS",
    "perplexity_search", "slack_health_check", "slack_token_status", "slack_users_info",
    "github_support_docs_search",
}


def is_read_only(name: str) -> bool:
    if name in _EXTRA_READS:
        return True
    lowered = name.lower()
    for prefix in ("googlecalendar_", "googledrive_", "gmail_", "github_", "linear_", "slack_"):
        lowered = lowered.removeprefix(prefix)
    return lowered.startswith(_READ_PREFIXES) or lowered.endswith(("_read", "_list", "_get", "_history"))


def _clean_args(args: dict[str, Any]) -> dict[str, Any]:
    """Strict mode makes the model send every parameter; null means "not set".
    Dropping nulls lets the tool's own defaults apply. This loses nothing: for a
    parameter whose default is None, omitting == passing None; for any other
    parameter, None isn't a valid value anyway. The cleaned dict is exactly what
    gets invoked and logged."""
    return {k: v for k, v in args.items() if v is not None}


def _is_deterministic(exc: Exception) -> bool:
    """Errors that will repeat on an identical retry: argument validation and
    the services' domain errors (all mock *MockError classes subclass
    ValueError). Anything else (timeouts, connection errors) may be transient,
    so an identical retry is still allowed."""
    return isinstance(exc, (ValidationError, ValueError, LookupError))


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
        self._failed: dict[str, str] = {}  # exact call -> deterministic error it produced

    def run(self, name: str, raw_args: dict[str, Any]) -> str:
        """Invoke a tool and return the text the model should see."""
        from backend.main import get_tool_spec

        args = _clean_args(raw_args)
        cache_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if is_read_only(name) and cache_key in self._read_cache:
            # Same read twice in one turn: answer from memory instead of re-calling.
            return self._read_cache[cache_key]
        if cache_key in self._failed:
            # This exact call already failed with an error that can't change on
            # retry (bad arguments, unknown ID). Not executed, so not logged.
            return json.dumps({"error": self._failed[cache_key],
                               "note": "This exact call already failed. Change the arguments or stop and explain."})

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
            if _is_deterministic(exc):
                self._failed[cache_key] = error
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
            # A write may change what reads return (and may fix what made a call
            # fail), so forget both. Writes themselves are never de-duplicated:
            # "post it twice" must post twice.
            self._read_cache.clear()
            self._failed.clear()
        return text

"""Runs registered tools and records every invocation for the response log."""
from __future__ import annotations

import json
import logging
import re
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


_DESTRUCTIVE = re.compile(r"delete|trash|remove|clear|empty", re.IGNORECASE)
_TRIVIAL_VALUES = {"me", "primary", "default"}
_LABEL_KEYS = ("name", "title", "summary", "subject", "snippet", "start", "from", "sender", "state")

_EMPTY_NOTE = ("Nothing matched with these arguments. Searches here match text literally, so retry with fewer, "
               "more distinctive words (e.g. 'board prep' rather than 'board prep meeting'). Before telling the user "
               "it doesn't exist, also consider another scope: another calendar (GOOGLECALENDAR_LIST_CALENDARS), "
               "shared drives, or another service.")


def is_destructive(name: str) -> bool:
    return bool(_DESTRUCTIVE.search(name))


def _item_lists(obj: Any, depth: int = 0):
    """Yield every list of 2+ records found in a tool result (a lookup's candidates)."""
    if depth > 3:
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _item_lists(value, depth + 1)
    elif isinstance(obj, list):
        records = [x for x in obj if isinstance(x, dict)]
        if len(records) >= 2:
            yield records
        for x in records:
            yield from _item_lists(x, depth + 1)


def _is_empty(result: Any) -> bool:
    """True when a read's result is only empty lists, e.g. {"events": [], "nextPageToken": null}."""
    if not isinstance(result, dict):
        return False
    lists = [v for v in result.values() if isinstance(v, list)]
    return bool(lists) and all(len(v) == 0 for v in lists)


def _label(record: dict[str, Any]) -> dict[str, Any]:
    out = {k: record[k] for k in ("id", "threadId") if k in record}
    for key in _LABEL_KEYS:
        value = record.get(key)
        if isinstance(value, (str, int)) or (isinstance(value, dict) and "dateTime" in value):
            out[key] = value if not isinstance(value, str) else value[:80]
    return out


class ToolExecutor:
    """Executes tools for one /chat turn and keeps the ordered call log."""

    def __init__(self) -> None:
        self.log: list[ToolCallLog] = []
        self._read_cache: dict[str, str] = {}
        self._failed: dict[str, str] = {}  # exact call -> deterministic error it produced
        self._recent_reads: list[Any] = []  # results of this turn's reads, newest last
        self._ambiguity_checked = False  # the destructive-action check fires at most once per turn

    def run(self, name: str, raw_args: dict[str, Any]) -> str:
        """Invoke a tool and return the text the model should see."""
        from backend.main import get_tool_spec

        args = _clean_args(raw_args)
        cache_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if is_read_only(name) and cache_key in self._read_cache:
            # Same read twice in one turn: answer from memory (not executed, not logged)
            # and say so, so the model doesn't keep asking for the same data.
            return ('{"note": "Same result as your earlier identical call - you already have this; '
                    'use it or try something different.", "result": ' + self._read_cache[cache_key] + "}")
        if cache_key in self._failed:
            # This exact call already failed with an error that can't change on
            # retry (bad arguments, unknown ID). Not executed, so not logged.
            return json.dumps({"error": self._failed[cache_key],
                               "note": "This exact call already failed. Change the arguments or stop and explain."})

        held = self._hold_ambiguous_destructive(name, args)
        if held is not None:
            return held  # not executed, so not logged

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
            self._recent_reads = [*self._recent_reads[-4:], result]
            # Echo the scope the read ran with (e.g. which calendar_id), so a later
            # write can use the right scope for an ID found here. Model view only.
            if _is_empty(result):
                text = json.dumps({"note": _EMPTY_NOTE, "called_with": args, "result": result}, default=str)
            elif len(text) < MAX_TOOL_OUTPUT_CHARS:
                text = '{"called_with": ' + json.dumps(args, default=str) + ', "result": ' + text + "}"
            self._read_cache[cache_key] = text
        else:
            # A write may change what reads return (and may fix what made a call
            # fail), so forget both. Writes themselves are never de-duplicated:
            # "post it twice" must post twice.
            self._read_cache.clear()
            self._failed.clear()
        return text

    def _hold_ambiguous_destructive(self, name: str, args: dict[str, Any]) -> str | None:
        """Execution-time check for destructive actions.

        If the target was picked out of a lookup in this turn that returned
        several records, hold the first destructive call and hand the
        candidates back: the model must repeat the call (the request clearly
        names this item) or ask the user. Unique targets pass straight through,
        and the check fires once per turn so "delete both" still works.
        """
        if self._ambiguity_checked or not is_destructive(name):
            return None
        targets = {v for v in args.values() if isinstance(v, str) and len(v) >= 3 and v not in _TRIVIAL_VALUES}
        if not targets:
            return None
        for result in reversed(self._recent_reads):
            for records in _item_lists(result):
                if any(targets & {v for v in r.values() if isinstance(v, str)} for r in records):
                    self._ambiguity_checked = True
                    log.info("held %s: target came from a lookup with %d matches", name, len(records))
                    return json.dumps({
                        "not_executed": True,
                        "reason": (f"Your lookup returned {len(records)} matching items and this {name} would act "
                                   "on one of them. If the user's request clearly identifies this exact item "
                                   "(e.g. by sender, date or exact name), repeat the same call to proceed. If it "
                                   "doesn't, don't guess: list the matches and ask the user which one."),
                        "matches": [_label(r) for r in records[:8]],
                    }, default=str)
        return None

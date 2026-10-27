"""Per-turn usage stats (LLM calls, tokens, timings) for logging and tests."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnStats:
    llm_calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    router_ms: float = 0.0
    total_ms: float = 0.0
    routed_tools: list[str] = field(default_factory=list)
    searched_tools: list[str] = field(default_factory=list)
    steps: int = 0

    def add_usage(self, response: Any) -> None:
        self.llm_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        details = getattr(usage, "input_tokens_details", None)
        self.cached_tokens += getattr(details, "cached_tokens", 0) or 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_current: ContextVar[TurnStats | None] = ContextVar("turn_stats", default=None)
LAST_TURN: dict[str, Any] = {}  # read by the scenario runner (single-threaded)


def start() -> TurnStats:
    stats = TurnStats()
    _current.set(stats)
    return stats


def current() -> TurnStats:
    return _current.get() or TurnStats()

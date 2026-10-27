"""Tool router: one small LLM call that picks the tools for this turn.

The router sees a one-line-per-tool catalog (~5k tokens, identical on every
call so OpenAI's prompt cache kicks in) plus the recent conversation, and
returns tool names constrained by a strict JSON-schema enum, so it can't
invent a tool. If the call fails we fall back to keyword search.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from backend.helpers import catalog, stats
from backend.helpers.config import MAX_ACTIVE_TOOLS, ROUTER_MODEL, get_client

log = logging.getLogger("chat.router")

_ROUTER_INSTRUCTIONS = """You are the tool router for an assistant that works across Gmail, Google Calendar, Google Drive, Slack, Linear, GitHub and Perplexity web search.

Read the conversation and choose every tool the assistant will need to fully handle the LATEST user message (use earlier messages to understand follow-ups like "the engineering one" or "yes, do it").

Rules:
- Cover the whole task: for "find X and post it to Slack", pick both the lookup tool and the write tool.
- Include the lookup tools needed to turn names into IDs (e.g. slack_list_conversations before slack_send_message, GOOGLEDRIVE_FIND_FILE before acting on a file, linear_list_teams / linear_list_projects / linear_list_users before creating or assigning Linear issues).
- Prefer the general list/search/fetch tool over narrow variants. Usually 1-8 tools are enough; never more than 15.
- If an item could live in more than one service (e.g. "the Launch Brief" could be a Drive file or an email; "the roadmap" a Drive doc or a Linear document), include the search tool for each plausible service.
- If the request names a service, stay inside it. Use perplexity_search only for questions about the outside world.
- Requests that are ambiguous still need the tools that would show the options (e.g. linear_list_projects for "the project").
- Return an empty list ONLY for greetings, small talk, or questions about yourself. Any request that touches the user's data or asks for an action gets tools, even if details are missing or it will need a clarifying question.

Tool catalog:
"""


@dataclass
class RouteResult:
    tools: list[str]
    source: str  # "llm" | "fallback"
    raw: list[str] = field(default_factory=list)


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": list(catalog.tool_names())},
            }
        },
        "required": ["tools"],
        "additionalProperties": False,
    }


def _transcript(history: list[dict[str, str]], max_messages: int = 6) -> str:
    lines = []
    for msg in history[-max_messages:]:
        content = msg["content"]
        if len(content) > 1500:
            content = content[:1500] + " ..."
        lines.append(f"{msg['role'].upper()}: {content}")
    return "\n\n".join(lines)


def route(history: list[dict[str, str]]) -> RouteResult:
    """Pick tools for the latest user turn. `history` holds role/content dicts."""
    try:
        response = get_client().responses.create(
            model=ROUTER_MODEL,
            instructions=_ROUTER_INSTRUCTIONS + catalog.compact_catalog(),
            input=_transcript(history),
            text={"format": {"type": "json_schema", "name": "tool_route", "schema": _schema(), "strict": True}},
            temperature=0,
            store=False,
        )
        stats.current().add_usage(response)
        picked = json.loads(response.output_text)["tools"]
        tools = catalog.with_companions(picked)[:MAX_ACTIVE_TOOLS]
        return RouteResult(tools=tools, source="llm", raw=picked)
    except Exception as exc:  # network, parsing, rate limits...
        log.warning("router failed, using keyword fallback: %s", exc)
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        tools = catalog.with_companions(catalog.keyword_search(last_user, limit=10))
        return RouteResult(tools=tools, source="fallback")

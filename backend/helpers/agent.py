"""Orchestration loop: route, then call the model with tools until it answers."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.helpers import catalog
from backend.helpers.config import AGENT_MODEL, MAX_ACTIVE_TOOLS, MAX_STEPS, get_client
from backend.helpers.executor import ToolExecutor
from backend.helpers.router import route

log = logging.getLogger("chat.agent")

SYSTEM_PROMPT = """You are a workplace assistant for Avery Quinn (Staff Engineer; Slack user U001, Linear user "me", avery@corp.com; Gmail account me@example.com). You act through tools over Gmail, Google Calendar, Google Drive, Slack, Linear, GitHub and Perplexity.

Current date and time: Wednesday, April 8, 2026, 09:00 America/New_York. Resolve relative dates ("tomorrow", "next week") from this; next week is Mon Apr 13 - Fri Apr 17, 2026. Use ISO 8601 datetimes.

How to work:
- Do exactly what was asked with as few tool calls as possible. Never repeat a lookup whose result you already have, and don't call tools for things you already know from the conversation.
- When several calls don't depend on each other, make them in the same step (parallel). Chain dependent calls in order.
- Resolve names to IDs with a lookup before writing (Slack channel names -> IDs via slack_list_conversations; files via GOOGLEDRIVE_FIND_FILE; Linear teams/projects/users via their list tools).
- Only fill the arguments the task needs. Leave optional fields (labels, priority, state, assignee, filters...) null unless the user asked for them or you got the value from a tool result - never invent IDs, labels or names.
- If you lack a needed tool, call search_tools with a short description of the capability.

Ambiguity:
- If the request could reasonably mean different things and guessing would change what gets written, sent, scheduled or deleted (which project, which channel, which of several matching items, missing recipient), ask ONE short clarifying question instead of acting. Offer the concrete options if you already know them or can get them with one cheap lookup.
- If the target named by the user doesn't exist (e.g. no Slack channel by that name), say so and list the closest options; do not substitute a different target on your own.
- Don't ask for confirmation of a clear, explicit instruction - just do it.

Honesty:
- Only report what tool results show. Never claim an action succeeded unless the tool returned success.
- If a search finds nothing or a tool errors, say so plainly and suggest a next step. If an error looks fixable (wrong ID format, missing argument), fix it and retry once.
- When asked to act on a specific item (file, email, event, issue), first check it exists. If it doesn't, say that clearly - that comes before anything else.
- Some actions have no tool (e.g. there is no tool to delete a Drive file). Say so instead of pretending.

Service tips:
- Gmail: GMAIL_FETCH_EMAILS returns only 1 message unless you set max_results (use ~10 for lists). The query matches every word as a substring of subject/body/sender, so use 1-2 short keywords (e.g. "timeline" not "timelines"); `from:`, `to:`, `subject:`, `label:`, `in:sent` work. Results are newest first. If nothing matches, retry once with a broader keyword.
- Drive: always call GOOGLEDRIVE_FIND_FILE with includeItemsFromAllDrives=true and corpora="allDrives" (some files live in shared drives). q looks like "name contains 'budget'"; search by a distinctive part of the name. Mention when a file is in the trash.
- Slack: slack_send_message needs a conversation ID from slack_list_conversations (C... channels, D... DMs, G... group DMs), never a user ID or channel name. To DM someone, find the DM conversation that has them as a member.
- Calendar: GOOGLECALENDAR_CREATE_EVENT takes start_datetime plus event_duration_hour/event_duration_minutes; attendees are email addresses.
- Linear: team arguments accept the team key (ENG, GRO) or name directly - no need to look teams up first. linear_list_projects shows each project's members.

Answer style: concise and specific. Summaries should use the real content from tool results. Mention IDs or names the user may need next."""

SEARCH_TOOLS = {
    "type": "function",
    "name": "search_tools",
    "description": (
        "Find and enable more tools when none of your current tools can do what is needed. "
        "Describe the capability, e.g. 'list github pull requests' or 'move calendar event'. "
        "Matching tools become callable on your next step."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Capability you need."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _to_history(messages: list[Any]) -> list[dict[str, str]]:
    history = []
    for m in messages:
        if m.role in ("user", "assistant"):
            history.append({"role": m.role, "content": m.content})
        elif m.role == "system":
            history.append({"role": "developer", "content": m.content})
        elif m.role == "tool":
            history.append({"role": "developer", "content": f"Tool output from an earlier turn:\n{m.content}"})
    return history


def _function_tools(names: list[str]) -> list[dict[str, Any]]:
    from backend.main import get_openai_tools

    return get_openai_tools(names) if names else []


def run_turn(messages: list[Any]) -> tuple[str, ToolExecutor]:
    executor = ToolExecutor()
    try:
        return _loop(messages, executor), executor
    except Exception as exc:
        exc.executor = executor  # so the caller can still return the partial log
        raise


def _loop(messages: list[Any], executor: ToolExecutor) -> str:
    started = time.perf_counter()
    history = _to_history(messages)
    routed = route([h for h in history if h["role"] in ("user", "assistant")])
    active = list(routed.tools)
    log.info("router(%s) -> %s (%.0fms)", routed.source, active, (time.perf_counter() - started) * 1000)

    client = get_client()
    items: list[Any] = list(history)

    for step in range(MAX_STEPS):
        response = client.responses.create(
            model=AGENT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=items,
            tools=[*_function_tools(active), SEARCH_TOOLS],
            parallel_tool_calls=True,
            temperature=0,
            store=False,
        )
        calls = [o for o in response.output if o.type == "function_call"]
        items.extend(o.model_dump(exclude_none=True) for o in response.output)
        if not calls:
            log.info("done in %d steps, %.0fms", step + 1, (time.perf_counter() - started) * 1000)
            return response.output_text.strip() or "Done."

        for call in calls:
            try:
                args = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                output = json.dumps({"error": "Arguments were not valid JSON."})
            else:
                if call.name == "search_tools":
                    output = _search_tools(args.get("query", ""), active)
                else:
                    output = executor.run(call.name, args)
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    # Step budget spent: ask for a final answer with tools switched off.
    log.warning("hit MAX_STEPS=%d, forcing a final answer", MAX_STEPS)
    response = client.responses.create(
        model=AGENT_MODEL,
        instructions=SYSTEM_PROMPT
        + "\n\nYou have run out of tool steps. Answer now with what you have, and be clear about anything left undone.",
        input=items,
        tools=[*_function_tools(active), SEARCH_TOOLS],
        tool_choice="none",
        temperature=0,
        store=False,
    )
    return response.output_text.strip() or "I couldn't finish this request."


def _search_tools(query: str, active: list[str]) -> str:
    found = [n for n in catalog.keyword_search(query, limit=6) if n not in active]
    room = MAX_ACTIVE_TOOLS + 8 - len(active)
    added = catalog.with_companions(found)[: max(room, 0)]
    active.extend(n for n in added if n not in active)
    log.info("search_tools(%r) -> %s", query, added)
    if not added:
        return json.dumps({"tools": [], "note": "No new matching tools. Your current tools may already cover this."})
    return json.dumps({"enabled": catalog.describe(added)})

"""Orchestration loop: route, then call the model with tools until it answers."""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from typing import Any

from backend.helpers import catalog, stats
from backend.helpers.config import AGENT_MODEL, MAX_ACTIVE_TOOLS, MAX_STEPS, get_client
from backend.helpers.executor import ToolExecutor
from backend.helpers.hints import allowed_values
from backend.helpers.router import route

log = logging.getLogger("chat.agent")

def _clock_line() -> str:
    """Today's date from the mock calendar clock, so the prompt can't drift from the fixtures."""
    from datetime import timedelta

    from backend.googlecalendar_mock.state import BASE_NOW, BASE_TIMEZONE

    monday = (BASE_NOW + timedelta(days=7 - BASE_NOW.weekday())).date()
    friday = monday + timedelta(days=4)
    return (
        f"Current date and time: {BASE_NOW:%A, %B %-d, %Y, %H:%M} {BASE_TIMEZONE} (UTC offset {BASE_NOW:%z}). "
        f"Resolve relative dates (\"today\", \"tomorrow\", \"next week\") from this; next week is "
        f"Mon {monday:%b %-d} - Fri {friday:%b %-d, %Y}. Use ISO 8601 datetimes with the offset."
    )


SYSTEM_PROMPT = """You are a workplace assistant for Avery Quinn (Staff Engineer; Slack user U001, Linear user "me", avery@corp.com; Gmail account me@example.com). You act through tools over Gmail, Google Calendar, Google Drive, Slack, Linear, GitHub and Perplexity.

{clock}

How to work:
- Do exactly what was asked with as few tool calls as possible. Never repeat a lookup whose result you already have, and don't call tools for things you already know from the conversation.
- When several calls don't depend on each other, make them in the same step (parallel). Chain dependent calls in order.
- Resolve names to IDs with a lookup before writing (Slack channel names -> IDs via slack_list_conversations; files via GOOGLEDRIVE_FIND_FILE; Linear teams/projects/users via their list tools).
- Only fill the arguments the task needs. Leave optional fields (labels, priority, state, assignee, filters...) null unless the user asked for them or you got the value from a tool result - never invent IDs, labels or names.
- If you lack a needed tool, call search_tools with a short description of the capability. Do this before asking the user for details you could look up yourself.
- Before saying something wasn't found, consider whether it could live in another service (a named brief or doc may be a Drive file rather than an email); if so, look there too, using search_tools to get the tool if needed.

Ambiguity:
- If the request could reasonably mean different things and guessing would change what gets written, sent, scheduled or deleted (which project, which channel, which of several matching items, missing recipient or content), ask ONE short clarifying question instead of acting. Do not start the task (checking availability, drafting, etc.) until it is resolved. At most one cheap list call to offer concrete options is fine.
- Destructive actions (delete, trash, cancel, remove): if your lookup returns more than one matching item - even two messages in the same thread - do NOT pick one. List the matches (subject/date/sender) and ask which one.
- If the target named by the user doesn't exist (e.g. no Slack channel by that name), say so and list the closest options; do not substitute a different target on your own.
- Don't ask for confirmation of a clear, explicit instruction - just do it. Not knowing the date or details of an item the user refers to ("my dentist appointment") is not ambiguity: look it up.

Honesty:
- Only report what tool results show. Never claim an action succeeded unless the tool returned success.
- If a search finds nothing or a tool errors, say so plainly and suggest a next step. If an error looks fixable (wrong ID format, missing argument), fix it and retry once.
- When checking whether something exists, don't narrow the lookup with optional filters (types, labels, states, date ranges) - an empty filtered result doesn't prove it's missing.
- When asked to act on a specific item (file, email, event, issue), first check it exists. If it doesn't, say that clearly - that comes before anything else.
- Some actions have no tool (e.g. there is no tool to delete a Drive file). Still look the item up first: if it doesn't exist, say that; if it exists, say the action isn't supported. Never pretend it was done.

Service tips:
- Gmail: "trash" or "move to trash" means GMAIL_MOVE_TO_TRASH (reversible); GMAIL_DELETE_MESSAGE is permanent, so use it only when the user asks to delete. GMAIL_FETCH_EMAILS returns only 1 message unless you set max_results (use ~10 for lists, and at least 5 when finding an email to delete, trash or reply to, so you can see whether several match). The query matches every word as a substring of subject/body/sender, so use 1-2 short keywords (e.g. "timeline" not "timelines"); `from:`, `to:`, `subject:`, `label:`, `in:sent` work. Results are newest first. If nothing matches, retry once with a broader keyword.
- Drive: always call GOOGLEDRIVE_FIND_FILE with includeItemsFromAllDrives=true and corpora="allDrives" (some files live in shared drives). q looks like "name contains 'budget'"; search by a distinctive part of the name. Mention when a file is in the trash.
- Slack: call slack_list_conversations without a types filter (what users call a "channel" may be a private channel or group DM). Post only to the conversation the user named - never to a different one. slack_send_message needs a conversation ID from slack_list_conversations (C... channels, D... DMs, G... group DMs), never a user ID or channel name. To DM someone, find the DM conversation that has them as a member.
- Calendar: GOOGLECALENDAR_FIND_EVENT matches query literally against event titles, so use 1-2 distinctive words ('board prep', 'dentist'), never extra words like 'meeting' or 'appointment'. To update or delete an event, pass the calendar_id of the calendar where you found it (not primary). The user has several calendars (primary plus secondary ones from GOOGLECALENDAR_LIST_CALENDARS). When looking for a specific event and it isn't on primary, search the other calendars too (in parallel) before saying it doesn't exist. GOOGLECALENDAR_CREATE_EVENT takes start_datetime plus event_duration_hour/event_duration_minutes; attendees are email addresses.
- Linear: team arguments accept the team key (ENG, GRO) or name directly - no need to look teams up first. Projects have no member list: "everyone on a project" means the members of the project's team (team memberIds from linear_list_teams / linear_get_team; emails from linear_list_users).

Answer style: concise and specific. Summaries should use the real content from tool results. When you found or changed specific items, name them with their IDs (e.g. thread thr_001, issue ENG-1, channel C001) so a follow-up message can refer to them without another lookup."""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{clock}", _clock_line())


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


def _nullable(prop: dict[str, Any]) -> dict[str, Any]:
    if prop.get("type") == "null" or any(o.get("type") == "null" for o in prop.get("anyOf", [])):
        return prop
    meta = {k: prop[k] for k in ("title", "description") if k in prop}
    inner = {k: v for k, v in prop.items() if k not in ("title", "description", "default")}
    if "default" in prop:
        meta["description"] = f"{meta.get('description', '')} Null = default ({prop['default']!r}).".strip()
    return {**meta, "anyOf": [inner, {"type": "null"}]}


@lru_cache(maxsize=256)
def _tool_schema(name: str) -> str:
    """OpenAI strict mode lists every property as required. For optional params
    that aren't nullable, that forces the model to invent a value (we saw fake
    pagination cursors and labels). Making them nullable keeps strict-mode
    guarantees while letting the model skip them; the executor drops nulls so
    the tool's own default applies."""
    from backend.main import get_openai_tools, get_tool_spec

    entry = get_openai_tools([name])[0]
    truly_required = set(get_tool_spec(name).args_model.model_json_schema().get("required", []))
    props = entry["parameters"].get("properties", {})
    for key, values in allowed_values(name).items():
        if key in props:
            props[key]["description"] = f"{props[key].get('description', '')} Allowed values: {', '.join(values)}.".strip()
    for key, prop in list(props.items()):
        if key not in truly_required:
            props[key] = _nullable(prop)
    return json.dumps(entry)


def _function_tools(names: list[str]) -> list[dict[str, Any]]:
    return [json.loads(_tool_schema(n)) for n in names]


def run_turn(messages: list[Any]) -> tuple[str, ToolExecutor]:
    executor = ToolExecutor()
    turn = stats.start()
    started = time.perf_counter()
    try:
        return _loop(messages, executor), executor
    except Exception as exc:
        exc.executor = executor  # so the caller can still return the partial log
        raise
    finally:
        turn.total_ms = (time.perf_counter() - started) * 1000
        summary = {**turn.as_dict(), "tool_calls": len(executor.log),
                   "tool_errors": sum(1 for c in executor.log if c.error)}
        stats.LAST_TURN.clear()
        stats.LAST_TURN.update(summary)
        log.info("turn %s", json.dumps(summary))


def _loop(messages: list[Any], executor: ToolExecutor) -> str:
    started = time.perf_counter()
    history = _to_history(messages)
    routed = route([h for h in history if h["role"] in ("user", "assistant")])
    active = list(routed.tools)
    stats.current().router_ms = (time.perf_counter() - started) * 1000
    stats.current().routed_tools = list(active)
    log.info("router(%s) -> %s (%.0fms)", routed.source, active, (time.perf_counter() - started) * 1000)

    client = get_client()
    items: list[Any] = list(history)

    idle_steps = 0
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
        stats.current().add_usage(response)
        stats.current().steps = step + 1
        calls = [o for o in response.output if o.type == "function_call"]
        items.extend(o.model_dump(exclude_none=True) for o in response.output)
        if not calls:
            log.info("done in %d steps, %.0fms", step + 1, (time.perf_counter() - started) * 1000)
            return response.output_text.strip() or "Done."

        executed_before = len(executor.log)
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

        # Loop guard: steps that only re-request results we already have (cache
        # hits / known failures) make no progress. Two in a row -> stop and answer.
        searched = any(c.name == "search_tools" for c in calls)
        idle_steps = 0 if (len(executor.log) > executed_before or searched) else idle_steps + 1
        if idle_steps >= 2:
            log.warning("no progress for %d steps, forcing a final answer", idle_steps)
            break

    # Step budget spent (or no progress): final answer with tools switched off.
    log.warning("stopping tool loop after %d steps, forcing a final answer", stats.current().steps)
    response = client.responses.create(
        model=AGENT_MODEL,
        instructions=SYSTEM_PROMPT
        + "\n\nStop calling tools now. Answer with what you have, and be clear about anything left undone.",
        input=items,
        tools=[*_function_tools(active), SEARCH_TOOLS],
        tool_choice="none",
        temperature=0,
        store=False,
    )
    stats.current().add_usage(response)
    return response.output_text.strip() or "I couldn't finish this request."


def _search_tools(query: str, active: list[str]) -> str:
    exact = [n for n in catalog.tool_names() if n.lower() in query.lower()]
    found = [n for n in [*exact, *catalog.keyword_search(query, limit=6)] if n not in active]
    room = MAX_ACTIVE_TOOLS + 8 - len(active)
    added = catalog.with_companions(found)[: max(room, 0)]
    active.extend(n for n in added if n not in active)
    stats.current().searched_tools.extend(added)
    log.info("search_tools(%r) -> %s", query, added)
    if not added:
        return json.dumps({"tools": [], "note": "No new matching tools. Your current tools may already cover this."})
    return json.dumps({"enabled": catalog.describe(added)})

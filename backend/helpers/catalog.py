"""Compact tool catalog, lookup companions and a keyword search over tools.

The catalog is the router's view of the world: one line per tool instead of
a full JSON schema, which keeps all 191 tools at roughly 5k tokens.
"""
from __future__ import annotations

import re
from functools import lru_cache

from backend.tooling import ToolSpec

SERVICE_LABELS = {
    "gmail": "Gmail",
    "googlecalendar": "Google Calendar",
    "googledrive": "Google Drive",
    "slack": "Slack",
    "linear": "Linear",
    "github": "GitHub",
    "perplexity": "Perplexity (web search)",
}

# Write tools need IDs that only a lookup tool can provide. Whenever the router
# picks the left side, we add the right side so the agent can resolve names.
COMPANIONS: dict[str, list[str]] = {
    "slack_send_message": ["slack_list_conversations"],
    "slack_conversations_history": ["slack_list_conversations"],
    "slack_get_full_conversation": ["slack_list_conversations"],
    "slack_get_thread": ["slack_conversations_history"],
    "slack_users_info": ["slack_list_users"],
    "GMAIL_REPLY_TO_THREAD": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_FETCH_MESSAGE_BY_THREAD_ID": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_MOVE_TO_TRASH": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_DELETE_MESSAGE": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_ADD_LABEL_TO_EMAIL": ["GMAIL_FETCH_EMAILS", "GMAIL_LIST_LABELS"],
    "GMAIL_GET_ATTACHMENT": ["GMAIL_FETCH_EMAILS"],
    "GOOGLEDRIVE_DOWNLOAD_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_EDIT_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_COPY_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_CREATE_COMMENT": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_ADD_FILE_SHARING_PREFERENCE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_CREATE_SHORTCUT_TO_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_FILES_MODIFY_LABELS": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLECALENDAR_UPDATE_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_PATCH_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_DELETE_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_REMOVE_ATTENDEE": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_EVENTS_MOVE": ["GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_LIST_CALENDARS"],
    "linear_create_issue": ["linear_list_teams"],
    "linear_update_issue": ["linear_list_issues"],
    "linear_create_comment": ["linear_list_issues"],
    "linear_update_project": ["linear_list_projects"],
    "linear_update_document": ["linear_list_documents"],
}

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "by", "with", "my", "me", "i"}


def _all_specs() -> dict[str, ToolSpec]:
    # Imported lazily: backend.main imports backend.solution, which imports us.
    from backend.main import get_all_tool_specs

    return get_all_tool_specs()


@lru_cache(maxsize=1)
def tool_names() -> tuple[str, ...]:
    return tuple(sorted(_all_specs()))


@lru_cache(maxsize=1)
def compact_catalog() -> str:
    by_service: dict[str, list[ToolSpec]] = {}
    for spec in _all_specs().values():
        by_service.setdefault(spec.service, []).append(spec)
    blocks = []
    for service, label in SERVICE_LABELS.items():
        specs = sorted(by_service.get(service, []), key=lambda s: s.name)
        lines = [f"- {s.name}: {s.description.strip()}" for s in specs]
        blocks.append(f"## {label}\n" + "\n".join(lines))
    return "\n\n".join(blocks)


# If a service is touched at all, its basic "find" tool comes along, so the
# agent can always check that a named item exists before acting on it.
SERVICE_ANCHORS = {"googledrive": "GOOGLEDRIVE_FIND_FILE"}


def with_companions(names: list[str]) -> list[str]:
    """Add lookup tools for any write/detail tool, preserving order and uniqueness."""
    specs = _all_specs()
    out: list[str] = []
    for name in names:
        if name not in specs:
            continue
        anchor = SERVICE_ANCHORS.get(specs[name].service)
        for candidate in [name, *COMPANIONS.get(name, []), *([anchor] if anchor else [])]:
            if candidate in specs and candidate not in out:
                out.append(candidate)
    return out


def _tokens(text: str) -> set[str]:
    words = set(_WORD.findall(text.lower().replace("_", " "))) - _STOP
    # crude stemming so "emails" matches "email", "issues" matches "issue"
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


@lru_cache(maxsize=1)
def _index() -> list[tuple[str, set[str]]]:
    return [
        (spec.name, _tokens(f"{spec.name} {spec.service} {SERVICE_LABELS.get(spec.service, '')} {spec.description}"))
        for spec in _all_specs().values()
    ]


def keyword_search(query: str, limit: int = 8) -> list[str]:
    """Rank tools by word overlap with the query. Used by the agent's
    `search_tools` escape hatch and as the router's fallback."""
    q = _tokens(query)
    if not q:
        return []
    scored = [(len(q & toks), name) for name, toks in _index()]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [name for _, name in scored[:limit]]


def describe(names: list[str]) -> list[dict[str, str]]:
    specs = _all_specs()
    return [{"name": n, "description": specs[n].description} for n in names if n in specs]

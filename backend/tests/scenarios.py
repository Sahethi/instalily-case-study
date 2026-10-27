"""Live scenario runner that mimics the grader (calls OpenAI, so not a unit test).

Run from the candidate/ directory:
    ./backend/.venv/bin/python -m backend.tests.scenarios            # all
    ./backend/.venv/bin/python -m backend.tests.scenarios s03 s07    # by id prefix
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import os

os.environ.setdefault("OPENAI_MAX_RETRIES", "8")  # personal keys can be limited to 10 RPM

from fastapi.testclient import TestClient

from backend.main import app


@dataclass
class Scenario:
    id: str
    messages: list[str] | list[dict[str, str]]
    required: list[str] = field(default_factory=list)       # tools that must be called
    forbidden: list[str] = field(default_factory=list)      # tools that must NOT be called
    min_calls: int | None = None                             # for the efficiency column
    reply_has: list[str] = field(default_factory=list)       # case-insensitive, any-of groups split by |
    state_check: Callable[[dict[str, Any]], bool] | None = None


def _seed_ts() -> set[str]:
    client = TestClient(app)
    client.post("/reset")
    return {m["ts"] for log in client.get("/state").json()["slack"]["channel_messages"] for m in log["messages"]}


_SEED_TS = _seed_ts()


def _slack_posted(channel: str, needle: str = "") -> Callable[[dict], bool]:
    def check(state: dict) -> bool:
        for log in state["slack"]["channel_messages"]:
            if log["channel"] == channel:
                return any(m["user"] == "U001" and needle.lower() in m["text"].lower() and m["ts"] not in _SEED_TS for m in log["messages"])
        return False
    return check


def _no_slack_post(state: dict) -> bool:
    return not any(m["ts"] not in _SEED_TS for log in state["slack"]["channel_messages"] for m in log["messages"])


def _linear_issue(title_part: str, team: str | None = None) -> Callable[[dict], bool]:
    def check(state: dict) -> bool:
        issues = state["linear"]["issues"]
        issues = issues.values() if isinstance(issues, dict) else issues
        return any(title_part.lower() in i["title"].lower() and (team is None or team in json.dumps(i)) for i in issues)
    return check


def _calendar_events(state: dict) -> int:
    return sum(len(c["events"]) for c in state["googlecalendar"]["events"])


SCENARIOS = [
    Scenario("s01_slack_list", ["What conversations do I have in Slack?"],
             required=["slack_list_conversations"], min_calls=1, reply_has=["engineering", "launch-war-room"]),
    Scenario("s02_greeting", ["Hello!"], forbidden=["*"], min_calls=0),
    Scenario("s03_email_to_slack", ["Find the most recent email about the Q1 revenue and post a summary to the engineering Slack channel."],
             required=["GMAIL_FETCH_EMAILS", "slack_send_message"], min_calls=3, state_check=_slack_posted("C001")),
    Scenario("s04_ambiguous_meeting", ["Schedule a 30-minute meeting with everyone on the project next week."],
             forbidden=["GOOGLECALENDAR_CREATE_EVENT"], min_calls=0, reply_has=["project|which"]),
    Scenario("s05_drive_missing", ["Delete the file 'budget_2025.xlsx' from my Drive."],
             required=["GOOGLEDRIVE_FIND_FILE"], min_calls=1, reply_has=["not|couldn|no file|doesn"]),
    Scenario("s06_missing_channel", ["Post 'standup moved to 10am' to the #timelines Slack channel."],
             required=["slack_list_conversations"], forbidden=["slack_send_message"], min_calls=1,
             reply_has=["timelines"], state_check=_no_slack_post),
    Scenario("s07_linear_issues", ["What Linear issues are on the Engineering team?"],
             required=["linear_list_issues"], min_calls=1, reply_has=["ENG-1|chat endpoint"]),
    Scenario("s08_perplexity_to_linear", ["Search the web for best practices on tool routing for LLM agents and create a Linear issue on the Engineering team summarizing them."],
             required=["perplexity_search", "linear_create_issue"], min_calls=2, state_check=_linear_issue("rout")),
    Scenario("s09_github_prs", ["List the open pull requests in instalily/orchestrator."],
             required=["github_list_pull_requests"], min_calls=1),
    Scenario("s10_calendar_today", ["What's on my calendar today?"], min_calls=1),
    Scenario("s11_followup", [
        {"role": "user", "content": "Schedule a 30-minute meeting with everyone on the project next week."},
        {"role": "assistant", "content": "Which project do you mean: Tool Orchestrator, Onboarding Refresh, or Renewals Risk?"},
        {"role": "user", "content": "Tool Orchestrator. Tuesday at 2pm works."},
    ], required=["GOOGLECALENDAR_CREATE_EVENT"], min_calls=2,
        state_check=lambda s: _calendar_events(s) > 3),
    Scenario("s12_drive_list", ["What files do I have in Google Drive?"], required=["GOOGLEDRIVE_FIND_FILE"], min_calls=1,
             reply_has=["Launch Brief"]),
    Scenario("s13_slack_dm", ["Send Morgan a Slack DM saying the revenue model is updated."],
             required=["slack_send_message"], min_calls=2, state_check=_slack_posted("D001", "revenue")),
    Scenario("s14_email_draft", ["Draft an email to finance@corp.com thanking them for the Q1 revenue review."],
             required=["GMAIL_CREATE_EMAIL_DRAFT"], min_calls=1,
             state_check=lambda s: len(s["gmail"]["drafts"]) > 1),
]


def run(scenario: Scenario) -> dict[str, Any]:
    client = TestClient(app)
    client.post("/reset")
    msgs = [m if isinstance(m, dict) else {"role": "user", "content": m} for m in scenario.messages]
    t0 = time.perf_counter()
    resp = client.post("/chat", json={"messages": msgs})
    ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    state = client.get("/state").json()
    calls = [c["name"] for c in body.get("tool_calls", [])]
    reply = body["messages"][-1]["content"] if resp.status_code == 200 else str(body)
    problems = []
    if resp.status_code != 200:
        problems.append(f"HTTP {resp.status_code}")
    if reply.startswith("Sorry, something went wrong"):
        problems.append("model/API error")
    for r in scenario.required:
        if r not in calls:
            problems.append(f"missing {r}")
    for f in scenario.forbidden:
        if (f == "*" and calls) or f in calls:
            problems.append(f"forbidden {f}")
    for group in scenario.reply_has:
        if not any(w.lower() in reply.lower() for w in group.split("|")):
            problems.append(f"reply lacks {group!r}")
    if scenario.state_check and not scenario.state_check(state):
        problems.append("state check failed")
    errors = [f'{c["name"]}({json.dumps(c["arguments"])[:200]}) -> {c["error"]}' for c in body.get("tool_calls", []) if c.get("error")]
    return {"id": scenario.id, "ok": not problems, "ms": ms, "calls": calls, "min": scenario.min_calls,
            "problems": problems, "reply": reply, "errors": errors}


def main(argv: list[str]) -> int:
    chosen = [s for s in SCENARIOS if not argv or any(s.id.startswith(a) for a in argv)]
    # TestClient shares one in-process mock state, so scenarios must run one at a time.
    with ThreadPoolExecutor(max_workers=1) as pool:
        results = list(pool.map(run, chosen))
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        extra = "" if r["min"] is None else f" ({len(r['calls'])}/{r['min']} calls)"
        print(f"{mark} [{r['id']}] {r['ms']:.0f}ms{extra} {r['calls']}" + (f" errors={r['errors']}" if r["errors"] else ""))
        if r["problems"]:
            print("     problems:", r["problems"])
        print("     reply:", r["reply"][:400].replace("\n", " "))
    passed = sum(r["ok"] for r in results)
    lat = sorted(r["ms"] for r in results)
    print(f"\nPass {passed}/{len(results)}  p50={lat[len(lat)//2]/1000:.1f}s  max={lat[-1]/1000:.1f}s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

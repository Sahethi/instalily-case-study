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
    required_any: list[str] = field(default_factory=list)   # at least one of these
    min_calls: int | None = None                             # for the efficiency column
    reply_has: list[str] = field(default_factory=list)       # case-insensitive, any-of groups split by |
    state_check: Callable[[dict[str, Any]], bool] | None = None
    force_route: list[str] | None = None  # override the router (forces mid-task search_tools discovery)
    turns: list[str] | None = None  # real multi-turn: each turn's reply feeds the next; checks apply to the last turn


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


def _dump(state: dict, service: str) -> str:
    return json.dumps(state[service]).lower()


def _new_slack_posts(state: dict, channel: str) -> int:
    return sum(1 for log in state["slack"]["channel_messages"] if log["channel"] == channel
               for m in log["messages"] if m["ts"] not in _SEED_TS)


def _gmail_msg(state: dict, msg_id: str) -> dict | None:
    return next((m for m in state["gmail"]["messages"] if m["id"] == msg_id), None)


def _gone(state: dict, msg_id: str) -> bool:
    m = _gmail_msg(state, msg_id)
    return m is None or "TRASH" in m.get("labelIds", [])


def _event_titles(state: dict) -> str:
    """Live events only (the state also keeps a change log that mentions deleted events)."""
    return " ".join(e.get("summary", "") for c in state["googlecalendar"]["events"] for e in c["events"]).lower()


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
    # ---- harder cases aimed at hidden-test categories ----
    Scenario("s15_reply_thread", ["Reply to the latest email from the CFO saying I'll have the summary ready by Friday."],
             required=["GMAIL_REPLY_TO_THREAD"], min_calls=2,
             state_check=lambda s: "friday" in _dump(s, "gmail")),
    Scenario("s16_ambiguous_delete", ["Delete the email about Q1 revenue."],
             forbidden=["GMAIL_DELETE_MESSAGE", "GMAIL_MOVE_TO_TRASH"], min_calls=1, reply_has=["which|both|two|2"]),
    Scenario("s17_tomorrow", ["What's on my calendar tomorrow?"], min_calls=1, reply_has=["deep work"]),
    Scenario("s18_linear_status", ["Move ENG-4 to In Progress."], required=["linear_update_issue"], min_calls=1,
             state_check=lambda s: any(i["id"] == "ENG-4" and "in_progress" in json.dumps(i) for i in s["linear"]["issues"])),
    Scenario("s19_github_issue", ["Open a GitHub issue in instalily/orchestrator titled 'Router misses Drive tools'."],
             required=["github_issue_write"], min_calls=1, state_check=lambda s: "router misses drive tools" in _dump(s, "github")),
    Scenario("s20_drive_read", ["What does the Launch Brief say?"], required=["GOOGLEDRIVE_FIND_FILE"], min_calls=2,
             reply_has=["launch"]),
    Scenario("s21_slack_to_draft", ["Summarize the recent messages in the #engineering Slack channel and save it as an email draft to morgan@corp.com."],
             required=["GMAIL_CREATE_EMAIL_DRAFT"], required_any=["slack_conversations_history", "slack_get_full_conversation"], min_calls=3,
             state_check=lambda s: "morgan@corp.com" in json.dumps(s["gmail"]["drafts"]).lower()),
    Scenario("s22_followup_it", [
        {"role": "user", "content": "What Linear issues are assigned to me?"},
        {"role": "assistant", "content": "You have 1 issue assigned to you: ENG-1 \"Wire /chat endpoint to orchestrator\" (In Progress, due Apr 11)."},
        {"role": "user", "content": "Add a comment to it saying I'm on it."},
    ], required=["linear_create_comment"], min_calls=1,
        state_check=lambda s: "on it" in json.dumps(s["linear"]["comments"]).lower()),
    Scenario("s23_missing_team", ["Create a Linear issue on the Marketing team titled 'Q2 campaign'."],
             reply_has=["marketing"], min_calls=1, state_check=lambda s: "q2 campaign" not in _dump(s, "linear")),
    Scenario("s24_web_search", ["Search the web for what MCP stands for in AI tooling."],
             required=["perplexity_search"], min_calls=1),
    Scenario("s25_free_time", ["When am I free tomorrow between 9am and 5pm?"], min_calls=1,
             required_any=["GOOGLECALENDAR_FIND_FREE_SLOTS", "GOOGLECALENDAR_FREE_BUSY_QUERY", "GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_FIND_EVENT"]),
    Scenario("s26_post_once", ["Post 'deploy finished' to #engineering."], required=["slack_send_message"], min_calls=2,
             state_check=lambda s: _new_slack_posts(s, "C001") == 1),
    Scenario("s27_github_to_slack", ["Check my GitHub notifications and post a summary in the launch-war-room Slack channel."],
             required=["github_list_notifications", "slack_send_message"], min_calls=3,
             state_check=lambda s: _new_slack_posts(s, "G001") == 1),
    Scenario("s28_cancel_dentist", ["Cancel my dentist appointment."], required=["GOOGLECALENDAR_DELETE_EVENT"], min_calls=2,
             state_check=lambda s: "dentist" not in _event_titles(s)),
    # ---- real multi-turn: earlier turns run through /chat, no reset between turns ----
    Scenario("m1_comment_on_first", [], turns=["What Linear issues are assigned to me?", "Add a comment to the first one saying I'm on it."],
             required=["linear_create_comment"], min_calls=1,
             state_check=lambda s: "on it" in json.dumps(s["linear"]["comments"]).lower()),
    Scenario("m2_reply_to_it", [], turns=["Find the most recent email from the CFO.", "Reply to it saying I'll have the numbers by Friday."],
             required=["GMAIL_REPLY_TO_THREAD"], min_calls=1, state_check=lambda s: "numbers by friday" in _dump(s, "gmail")),
    Scenario("m3_post_engineering_one", [], turns=["What conversations do I have in Slack?", "Post 'hello team' in the engineering one."],
             required=["slack_send_message"], min_calls=1, state_check=lambda s: _new_slack_posts(s, "C001") == 1),
    Scenario("m4_clarify_then_schedule", [], turns=["Schedule a 30-minute meeting with everyone on the project next week.",
                                                     "The Tool Orchestrator project, Tuesday at 2pm."],
             required=["GOOGLECALENDAR_CREATE_EVENT"], min_calls=2, state_check=lambda s: _calendar_events(s) > 3),
    Scenario("m5_read_listed_file", [], turns=["What files do I have in Google Drive?", "What does the Launch Brief say?"],
             required=["GOOGLEDRIVE_DOWNLOAD_FILE"], min_calls=1),
    Scenario("m6_summarize_channel_by_name", [], turns=["What conversations do I have in Slack?",
                                                         "Summarize the engineering channel and save it as an email draft to morgan@corp.com"],
             required=["GMAIL_CREATE_EMAIL_DRAFT"], required_any=["slack_conversations_history", "slack_get_full_conversation"], min_calls=3,
             state_check=lambda s: "morgan@corp.com" in json.dumps(s["gmail"]["drafts"]).lower()),
    # ---- destructive actions that MUST still execute (guards against over-blocking) ----
    Scenario("d1_delete_specific_match", ["Delete the email from the CFO about Q1 revenue."],
             required_any=["GMAIL_DELETE_MESSAGE", "GMAIL_MOVE_TO_TRASH"], min_calls=2,
             state_check=lambda s: _gone(s, "msg_002") and not _gone(s, "msg_001")),
    Scenario("d2_trash_unique", ["Trash the office move checklist email."], required=["GMAIL_MOVE_TO_TRASH"], min_calls=2,
             state_check=lambda s: _gone(s, "msg_006")),
    Scenario("d3_cancel_primary_event", ["Cancel the Revenue board prep meeting."], required=["GOOGLECALENDAR_DELETE_EVENT"],
             min_calls=2, state_check=lambda s: "revenue board prep" not in _event_titles(s)),
    # ---- router deliberately incomplete: the agent must discover tools with search_tools ----
    Scenario("t1_discover_send", ["Post 'hello' to #engineering."], force_route=["slack_list_conversations"],
             required=["slack_send_message"], min_calls=2, state_check=lambda s: _new_slack_posts(s, "C001") == 1),
    Scenario("t2_discover_cross_service", ["Find the most recent email about the product launch and post a summary to #engineering."],
             force_route=["GMAIL_FETCH_EMAILS"], required=["GMAIL_FETCH_EMAILS", "slack_send_message"], min_calls=3,
             state_check=lambda s: _new_slack_posts(s, "C001") == 1),
]


def run(scenario: Scenario) -> dict[str, Any]:
    client = TestClient(app)
    client.post("/reset")
    msgs = [m if isinstance(m, dict) else {"role": "user", "content": m} for m in scenario.messages]
    earlier_calls: list[str] = []
    if scenario.turns:
        for text in scenario.turns[:-1]:
            prior = client.post("/chat", json={"messages": [*msgs, {"role": "user", "content": text}]}).json()
            msgs = prior["messages"]
            earlier_calls += [c["name"] for c in prior["tool_calls"]]
        msgs = [*msgs, {"role": "user", "content": scenario.turns[-1]}]
    t0 = time.perf_counter()
    if scenario.force_route:
        from unittest import mock

        from backend.helpers.router import RouteResult
        with mock.patch("backend.helpers.agent.route", return_value=RouteResult(tools=list(scenario.force_route), source="forced")):
            resp = client.post("/chat", json={"messages": msgs})
    else:
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
    if scenario.required_any and not any(r in calls for r in scenario.required_any):
        problems.append(f"none of {scenario.required_any}")
    for f in scenario.forbidden:
        if (f == "*" and calls) or f in calls:
            problems.append(f"forbidden {f}")
    for group in scenario.reply_has:
        if not any(w.lower() in reply.lower() for w in group.split("|")):
            problems.append(f"reply lacks {group!r}")
    if scenario.state_check and not scenario.state_check(state):
        problems.append("state check failed")
    from backend.helpers import stats as turn_stats
    usage = dict(turn_stats.LAST_TURN)
    errors = [f'{c["name"]}({json.dumps(c["arguments"])[:200]}) -> {c["error"]}' for c in body.get("tool_calls", []) if c.get("error")]
    return {"id": scenario.id, "ok": not problems, "ms": ms, "calls": calls, "min": scenario.min_calls,
            "problems": problems, "reply": reply, "errors": errors, "usage": usage,
            "earlier_turn_calls": earlier_calls,
            "turn1_reply": msgs[-2]["content"] if scenario.turns else None,
            "tool_log": [{"name": c["name"], "arguments": c["arguments"], "error": c.get("error")} for c in body.get("tool_calls", [])]}


def main(argv: list[str]) -> int:
    label = next((a.split("=", 1)[1] for a in argv if a.startswith("--label=")), None)
    if "--no-id-hint" in argv:  # A/B: restore the reply-style sentence used before the ID instruction
        from backend.helpers import agent
        head, _, _ = agent.SYSTEM_PROMPT.rpartition("When you found or changed specific items")
        agent.SYSTEM_PROMPT = head + "Mention IDs or names the user may need next."
    argv = [a for a in argv if not a.startswith("--")]
    chosen = [s for s in SCENARIOS if not argv or any(s.id.startswith(a) for a in argv)]
    # TestClient shares one in-process mock state, so scenarios must run one at a time.
    with ThreadPoolExecutor(max_workers=1) as pool:
        results = list(pool.map(run, chosen))
    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        extra = "" if r["min"] is None else f" ({len(r['calls'])}/{r['min']} calls)"
        u = r["usage"]
        print(f"{mark} [{r['id']}] {r['ms']:.0f}ms{extra} llm={u.get('llm_calls')} tok={u.get('input_tokens')}/{u.get('cached_tokens')}c/{u.get('output_tokens')}o {r['calls']}" + (f" errors={r['errors']}" if r["errors"] else ""))
        if r["problems"]:
            print("     problems:", r["problems"])
        print("     reply:", r["reply"][:400].replace("\n", " "))
    passed = sum(r["ok"] for r in results)
    over = [(r["id"], len(r["calls"]) - r["min"]) for r in results if r["min"] is not None and len(r["calls"]) > r["min"]]
    tot = lambda k: sum(r["usage"].get(k) or 0 for r in results)
    print(f"Tool calls: {sum(len(r['calls']) for r in results)} total, {sum(len(r['errors']) for r in results)} errored, "
          f"{len(over)} scenarios over minimum (+{sum(x for _, x in over)})")
    print(f"LLM calls: {tot('llm_calls')}  input tokens: {tot('input_tokens')} (cached {tot('cached_tokens')})  output tokens: {tot('output_tokens')}")
    if label:
        out = os.path.join(os.environ.get("SCENARIO_OUT", "."), f"scenarios_{label}.json")
        with open(out, "w") as fh:
            json.dump(results, fh, indent=1, default=str)
        print("saved", out)
    lat = sorted(r["ms"] for r in results)
    print(f"\nPass {passed}/{len(results)}  p50={lat[len(lat)//2]/1000:.1f}s  max={lat[-1]/1000:.1f}s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

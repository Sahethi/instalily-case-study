"""Deterministic tests for the executor, schemas and router fallback (no OpenAI calls)."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from backend.helpers import agent, router
from backend.helpers.executor import ToolExecutor
from backend.main import get_tool_spec, reset_all_mock_state
from backend.slack_mock.state import get_state as slack_state
from backend.tooling import ToolSpec


def _engineering_posts(text: str) -> int:
    return sum(1 for log in slack_state().channel_messages if log.channel == "C001" for m in log.messages if m.text == text)


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_all_mock_state()
        self.ex = ToolExecutor()

    def test_nulls_dropped_and_logged_args_equal_invoked_args(self) -> None:
        invoked: list[dict] = []
        original = ToolSpec.invoke

        def spy(spec_self, **kwargs):
            invoked.append(dict(kwargs))
            return original(spec_self, **kwargs)

        with mock.patch.object(ToolSpec, "invoke", spy):
            self.ex.run("slack_send_message", {"channel": "C001", "text": "hi", "thread_ts": None})
        self.assertEqual(invoked, [{"channel": "C001", "text": "hi"}])
        self.assertEqual(self.ex.log[0].arguments, invoked[0])

    def test_null_for_non_nullable_default_falls_back_to_tool_default(self) -> None:
        # linear_list_projects.limit is `int = 50`; null must not reach the tool.
        self.ex.run("linear_list_projects", {"limit": None, "orderBy": None})
        self.assertIsNone(self.ex.log[0].error)
        self.assertEqual(self.ex.log[0].arguments, {})

    def test_intentional_repeated_writes_both_execute_and_log(self) -> None:
        for _ in range(2):
            self.ex.run("slack_send_message", {"channel": "C001", "text": "ping"})
        self.assertEqual([c.name for c in self.ex.log], ["slack_send_message"] * 2)
        self.assertEqual(_engineering_posts("ping"), 2)

    def test_identical_deterministic_failure_not_reexecuted_or_logged(self) -> None:
        first = json.loads(self.ex.run("slack_send_message", {"channel": "NOPE", "text": "x"}))
        second = json.loads(self.ex.run("slack_send_message", {"channel": "NOPE", "text": "x"}))
        self.assertIn("error", first)
        self.assertIn("already failed", second["note"])
        self.assertEqual(len(self.ex.log), 1)  # log holds only real executions

    def test_transient_failure_can_be_retried(self) -> None:
        spec = get_tool_spec("slack_list_conversations")
        calls = {"n": 0}

        def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("upstream timed out")
            return spec.invoke(**kwargs)

        fake = mock.Mock(invoke=flaky)
        with mock.patch("backend.main.get_tool_spec", return_value=fake):
            self.assertIn("error", json.loads(self.ex.run("slack_list_conversations", {})))
            self.assertNotIn("error", json.loads(self.ex.run("slack_list_conversations", {})))
        self.assertEqual(calls["n"], 2)
        self.assertEqual([c.error is None for c in self.ex.log], [False, True])

    def test_read_cache_hit_not_logged_and_cleared_by_write(self) -> None:
        self.ex.run("slack_list_conversations", {})
        self.ex.run("slack_list_conversations", {})  # cache hit, not executed
        self.ex.run("slack_send_message", {"channel": "C001", "text": "x"})
        self.ex.run("slack_list_conversations", {})  # after a write: executed again
        self.assertEqual([c.name for c in self.ex.log],
                         ["slack_list_conversations", "slack_send_message", "slack_list_conversations"])

    def test_failed_call_memo_cleared_by_write(self) -> None:
        self.ex.run("slack_get_thread", {"channel": "C001", "thread_ts": "1"})
        self.ex.run("slack_send_message", {"channel": "C001", "text": "x"})
        self.ex.run("slack_get_thread", {"channel": "C001", "thread_ts": "1"})
        self.assertEqual(len(self.ex.log), 3)


class SchemaTests(unittest.TestCase):
    def test_optional_params_nullable_required_untouched(self) -> None:
        params = agent._function_tools(["linear_list_projects"])[0]["parameters"]
        self.assertIn({"type": "null"}, params["properties"]["limit"]["anyOf"])
        from backend.main import get_openai_tools

        ours = agent._function_tools(["linear_create_issue"])[0]["parameters"]["properties"]
        original = get_openai_tools(["linear_create_issue"])[0]["parameters"]["properties"]
        for required in ("title", "team"):  # required params are left exactly as generated
            self.assertEqual(ours[required], original[required])

    def test_allowed_values_from_source(self) -> None:
        method = agent._function_tools(["github_issue_write"])[0]["parameters"]["properties"]["method"]
        self.assertIn("Allowed values: create, update", method["description"])


class RouterFallbackTests(unittest.TestCase):
    def test_router_failure_falls_back_to_keyword_search(self) -> None:
        with mock.patch.object(router, "get_client", side_effect=RuntimeError("API down")):
            result = router.route([{"role": "user", "content": "send a slack message to the engineering channel"}])
        self.assertEqual(result.source, "fallback")
        self.assertIn("slack_send_message", result.tools)
        self.assertIn("slack_list_conversations", result.tools)


if __name__ == "__main__":
    unittest.main()


# ---- loop control and failure paths, driven by a scripted fake model ----
from types import SimpleNamespace  # noqa: E402

from backend.chat_schema import ChatMessage, ChatRequest  # noqa: E402
from backend.helpers.config import MAX_STEPS  # noqa: E402
from backend.helpers.router import RouteResult  # noqa: E402


def _call(name: str, args: dict, call_id: str):
    item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id)
    item.model_dump = lambda **_: {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}
    return item


def _response(output=(), text=""):
    return SimpleNamespace(output=list(output), output_text=text, usage=None)


class FakeModel:
    """Plays back a script; each entry is a response or an exception to raise."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0) if self.script else _response(text="final answer")
        if isinstance(step, Exception):
            raise step
        return step


class LoopControlTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_all_mock_state()
        self.route = mock.patch.object(agent, "route", return_value=RouteResult(tools=["slack_list_conversations", "slack_send_message"], source="test"))
        self.route.start()

    def tearDown(self) -> None:
        self.route.stop()

    def _chat(self, fake):
        from backend.solution import chat

        with mock.patch.object(agent, "get_client", return_value=fake):
            return chat(ChatRequest(messages=[ChatMessage(role="user", content="do it")]))

    def test_step_limit_forces_final_answer_without_tools(self) -> None:
        # Every step asks for a *different* read, so it always makes progress.
        fake = FakeModel([_response([_call("slack_list_conversations", {"limit": i + 1}, f"c{i}")]) for i in range(MAX_STEPS)])
        out = self._chat(fake)
        self.assertEqual(len(out.tool_calls), MAX_STEPS)
        self.assertEqual(len(fake.requests), MAX_STEPS + 1)
        self.assertEqual(fake.requests[-1]["tool_choice"], "none")
        self.assertEqual(out.messages[-1].content, "final answer")

    def test_no_progress_guard_stops_repeated_identical_reads(self) -> None:
        fake = FakeModel([_response([_call("slack_list_conversations", {}, f"c{i}")]) for i in range(MAX_STEPS)])
        out = self._chat(fake)
        self.assertEqual(len(out.tool_calls), 1)  # executed once; repeats were cache hits
        self.assertEqual(len(fake.requests), 4)  # 1 real + 2 idle steps, then forced answer

    def test_model_failure_mid_request_keeps_partial_log(self) -> None:
        fake = FakeModel([_response([_call("slack_send_message", {"channel": "C001", "text": "hi"}, "c1")]),
                          RuntimeError("model API down")])
        out = self._chat(fake)
        self.assertEqual([c.name for c in out.tool_calls], ["slack_send_message"])  # side effect is reported
        self.assertIn("went wrong", out.messages[-1].content)

    def test_parallel_calls_from_one_step_all_execute_in_order(self) -> None:
        fake = FakeModel([_response([_call("slack_list_conversations", {}, "a"),
                                     _call("slack_send_message", {"channel": "C001", "text": "x"}, "b")])])
        out = self._chat(fake)
        self.assertEqual([c.name for c in out.tool_calls], ["slack_list_conversations", "slack_send_message"])


class DestructiveCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_all_mock_state()
        self.ex = ToolExecutor()

    def test_target_from_multi_match_lookup_is_held_once_then_runs(self) -> None:
        self.ex.run("GMAIL_FETCH_EMAILS", {"query": "Q1 revenue", "max_results": 5})
        held = json.loads(self.ex.run("GMAIL_DELETE_MESSAGE", {"message_id": "msg_001"}))
        self.assertTrue(held["not_executed"])
        self.assertEqual({m["id"] for m in held["matches"]}, {"msg_001", "msg_002"})
        self.assertEqual([c.name for c in self.ex.log], ["GMAIL_FETCH_EMAILS"])  # held call not logged
        self.ex.run("GMAIL_DELETE_MESSAGE", {"message_id": "msg_001"})
        self.assertEqual([c.name for c in self.ex.log], ["GMAIL_FETCH_EMAILS", "GMAIL_DELETE_MESSAGE"])

    def test_unique_target_runs_immediately(self) -> None:
        self.ex.run("GMAIL_FETCH_EMAILS", {"query": "office move", "max_results": 5})
        self.ex.run("GMAIL_MOVE_TO_TRASH", {"message_id": "msg_006"})
        self.assertEqual([c.name for c in self.ex.log], ["GMAIL_FETCH_EMAILS", "GMAIL_MOVE_TO_TRASH"])

    def test_non_destructive_write_never_held(self) -> None:
        self.ex.run("slack_list_conversations", {})
        self.ex.run("slack_send_message", {"channel": "C001", "text": "x"})
        self.assertEqual(len(self.ex.log), 2)

    def test_empty_read_gets_scope_note(self) -> None:
        out = json.loads(self.ex.run("GOOGLECALENDAR_FIND_EVENT", {"query": "dentist"}))
        self.assertIn("another calendar", out["note"])
        self.assertEqual(out["called_with"], {"query": "dentist"})
        self.assertEqual(self.ex.log[0].result["events"], [])  # log keeps the raw result

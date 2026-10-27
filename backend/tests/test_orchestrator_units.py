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

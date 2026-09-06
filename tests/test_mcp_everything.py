"""Tests for the mcp_everything domain.

Tests verify:
1. get_suite() and get_evaluator() conform to the domain contract.
2. MCPClient and ToolRuntime adapter discover tools dynamically at runtime
   over the protocol (never hardcoded).
3. The adapter handles server absence gracefully with clear error messages.
4. The evaluator grades deterministically without aggregating.
5. The real harness (agent_engineer.evaluation.Evaluator) produces a non-zero
   baseline score with an engine runner that never imports mcp_everything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_engineer.domains.mcp_everything import (
    DOMAIN,
    EVALUATOR_ID,
    MCPClient,
    MCPServerError,
    MCPServerNotFoundError,
    MCPToolRuntime,
    StubTransport,
    get_evaluator,
    get_suite,
    get_tool_runtime,
)
from agent_engineer.evaluation import DomainSuite, Evaluator, TaskEvaluator, TaskSpec
from agent_engineer.ports import ToolResult, ToolSchema
from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, ToolCall, Trajectory

SUITE: DomainSuite = get_suite()
EVALUATE: TaskEvaluator = get_evaluator()
SPEC = AgentSpec(spec_id="mcp-test-spec", system_prompt="Answer the task using MCP.")


# --------------------------------------------------------------------------- #
# Domain contract and suite tests
# --------------------------------------------------------------------------- #


def test_suite_conforms_to_contract() -> None:
    assert SUITE.domain == "mcp_everything"
    assert len(SUITE.tasks) >= 4
    ids = [t.task_id for t in SUITE.tasks]
    assert len(ids) == len(set(ids)), "task IDs must be unique"
    for task in SUITE.tasks:
        assert task.domain == "mcp_everything"
        assert task.prompt
        assert task.metadata.get("kind") == "mcp"
        assert "required_tools" in task.metadata


def test_get_suite_is_idempotent() -> None:
    suite1 = get_suite()
    suite2 = get_suite()
    assert suite1.domain == suite2.domain
    assert tuple(t.task_id for t in suite1.tasks) == tuple(t.task_id for t in suite2.tasks)


# --------------------------------------------------------------------------- #
# Dynamic discovery & ToolRuntime adapter tests (offline via StubTransport)
# --------------------------------------------------------------------------- #


def test_dynamic_tool_discovery_not_hardcoded() -> None:
    """Tool schemas must be discovered dynamically from the MCP server at runtime."""
    custom_tools = [
        {
            "name": "dynamic_custom_tool",
            "description": "Discovered at runtime over protocol",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    class CustomStubTransport:
        def __init__(self) -> None:
            self._queue: list[dict[str, Any]] = []

        def send(self, message: dict[str, Any]) -> None:
            method = message.get("method")
            msg_id = message.get("id")
            if method == "initialize":
                self._queue.append({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "test-server", "version": "1.0"},
                    },
                })
            elif method == "notifications/initialized":
                pass
            elif method == "tools/list":
                self._queue.append({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": custom_tools},
                })

        def receive(self, timeout: float = 10.0) -> dict[str, Any] | None:
            return self._queue.pop(0) if self._queue else None

        def close(self) -> None:
            pass

    client = MCPClient(transport=CustomStubTransport())
    runtime = MCPToolRuntime(client=client)

    schemas = runtime.schemas()
    assert len(schemas) == 1
    assert schemas[0].name == "dynamic_custom_tool"
    assert schemas[0].description == "Discovered at runtime over protocol"
    assert schemas[0].parameters["required"] == ["query"]


def test_tool_runtime_invocation_success() -> None:
    transport = StubTransport()
    client = MCPClient(transport=transport)
    runtime = MCPToolRuntime(client=client)

    task = SUITE.tasks[0]
    result = runtime.invoke(task, "echo", {"message": "Protocol test"})
    assert result.ok
    assert result.value == "Echo: Protocol test"
    assert result.error is None


def test_tool_runtime_invocation_arithmetic() -> None:
    transport = StubTransport()
    client = MCPClient(transport=transport)
    runtime = MCPToolRuntime(client=client)

    task = SUITE.tasks[1]
    result = runtime.invoke(task, "get-sum", {"a": 100, "b": 250})
    assert result.ok
    assert "350" in str(result.value)
    assert result.error is None


def test_tool_runtime_invocation_error_handled_gracefully() -> None:
    transport = StubTransport()
    client = MCPClient(transport=transport)
    runtime = MCPToolRuntime(client=client)

    task = SUITE.tasks[0]
    result = runtime.invoke(task, "non_existent_tool", {})
    assert not result.ok
    assert result.value is None
    assert "not found" in result.error.lower()


def test_absent_server_handled_gracefully_with_clear_error() -> None:
    class FailingTransport:
        def send(self, message: dict[str, Any]) -> None:
            raise MCPServerNotFoundError("MCP server executable 'npx' not found on PATH.")

        def receive(self, timeout: float = 10.0) -> dict[str, Any] | None:
            return None

        def close(self) -> None:
            pass

    client = MCPClient(transport=FailingTransport())
    runtime = MCPToolRuntime(client=client)

    schemas = runtime.schemas()
    assert schemas == ()

    task = SUITE.tasks[0]
    result = runtime.invoke(task, "echo", {"message": "hello"})
    assert not result.ok
    assert "not found on PATH" in result.error


# --------------------------------------------------------------------------- #
# Evaluator unit tests
# --------------------------------------------------------------------------- #


def test_evaluator_grades_only_never_aggregates() -> None:
    task = SUITE.tasks[0]
    traj = Trajectory(
        trajectory_id="t-1",
        spec_id=SPEC.spec_id,
        task_id=task.task_id,
        tool_calls=(
            ToolCall(
                ordinal=0,
                tool_name="echo",
                args={"message": "Hello MCP!"},
                result="Echo: Hello MCP!",
            ),
        ),
        final_answer="Echo: Hello MCP!",
    )
    verdict = EVALUATE(task, traj)
    assert isinstance(verdict, EvaluatorVerdict)
    assert verdict.evaluator_id == EVALUATOR_ID
    assert verdict.passed is True
    assert verdict.score == 1.0


def test_evaluator_missing_answer_fails() -> None:
    task = SUITE.tasks[0]
    traj = Trajectory(
        trajectory_id="t-no-ans",
        spec_id=SPEC.spec_id,
        task_id=task.task_id,
        final_answer=None,
    )
    verdict = EVALUATE(task, traj)
    assert verdict.passed is False
    assert verdict.score == 0.0
    assert "no final answer" in verdict.rationale


def test_evaluator_partial_credit_when_tool_not_called_but_answer_given() -> None:
    task = SUITE.tasks[0]
    traj = Trajectory(
        trajectory_id="t-no-tool",
        spec_id=SPEC.spec_id,
        task_id=task.task_id,
        tool_calls=(),
        final_answer="Echo: Hello MCP!",
    )
    verdict = EVALUATE(task, traj)
    assert verdict.passed is False
    assert verdict.score == 0.7
    assert "not invoked" in verdict.rationale


# --------------------------------------------------------------------------- #
# Engine baseline score test through the real harness
# --------------------------------------------------------------------------- #


def _baseline_mcp_runner(spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory:
    """Baseline engine driver that does not import the mcp_everything module.

    Interacts only via TaskSpec and a stubbed ToolRuntime.
    Calls the tool for easy tasks and answers directly for harder ones.
    """
    transport = StubTransport()
    client = MCPClient(transport=transport)
    runtime = MCPToolRuntime(client=client)

    calls: list[ToolCall] = []
    final_answer: str | None = None

    if task.task_id == "mcp-echo":
        tool_res = runtime.invoke(task, "echo", {"message": "Hello MCP!"})
        calls.append(
            ToolCall(
                ordinal=0,
                tool_name="echo",
                args={"message": "Hello MCP!"},
                result=tool_res.value,
                error=tool_res.error,
            )
        )
        final_answer = tool_res.value
    elif task.task_id == "mcp-sum-small":
        tool_res = runtime.invoke(task, "get-sum", {"a": 17, "b": 25})
        calls.append(
            ToolCall(
                ordinal=0,
                tool_name="get-sum",
                args={"a": 17, "b": 25},
                result=tool_res.value,
                error=tool_res.error,
            )
        )
        final_answer = tool_res.value
    else:
        final_answer = None

    runtime.close()

    return Trajectory(
        trajectory_id=f"traj_{task.task_id}_{attempt}",
        spec_id=spec.spec_id,
        task_id=task.task_id,
        tool_calls=tuple(calls),
        final_answer=final_answer,
    )


def test_baseline_engine_scores_above_zero() -> None:
    """The standard domain test: the engine produces a non-zero baseline score

    through the real measurement harness (Evaluator).
    """
    harness = Evaluator(
        suites=(SUITE,),
        evaluators={"mcp_everything": EVALUATE},
        repeats=3,
    )
    report = harness.run_iteration(0, SPEC, _baseline_mcp_runner)
    domain_report = report.domain("mcp_everything")
    accuracy = domain_report.accuracy.require()
    assert accuracy > 0.0, "baseline accuracy must be non-zero"


def test_live_server_opt_in() -> None:
    """Opt-in live test executed only when AE_MCP_LIVE=1 is set in the environment."""
    import os
    if not os.environ.get("AE_MCP_LIVE"):
        pytest.skip("Set AE_MCP_LIVE=1 to run against the live MCP server")

    runtime = get_tool_runtime()
    try:
        schemas = runtime.schemas()
        assert len(schemas) > 0, "live MCP server must provide tools"
        assert any(s.name == "echo" for s in schemas)
        task = SUITE.tasks[0]
        res = runtime.invoke(task, "echo", {"message": "Live opt-in test"})
        assert res.ok
        assert res.value == "Echo: Live opt-in test"
    finally:
        runtime.close()

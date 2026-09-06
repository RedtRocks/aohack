"""The task catalogue for the mcp_everything domain.

Exercises the real MCP reference server tools: echo, get-sum, get-annotated-message,
get-env, and get-tiny-image. Each task requires discovering and executing tools over
the Model Context Protocol stdio transport.
"""

from __future__ import annotations

from typing import Any

from agent_engineer.evaluation import DomainSuite, TaskSpec

__all__ = ["DOMAIN", "load_suite", "build_suite"]

DOMAIN = "mcp_everything"


def _mcp_task(
    task_id: str,
    *,
    difficulty: str,
    prompt: str,
    expected: str,
    required_tools: tuple[str, ...],
    expected_args: dict[str, Any],
    expected_fragment: str,
    trap: str,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        domain=DOMAIN,
        prompt=prompt,
        expected=expected,
        metadata={
            "kind": "mcp",
            "difficulty": difficulty,
            "required_tools": list(required_tools),
            "expected_args": expected_args,
            "expected_fragment": expected_fragment,
            "trap": trap,
        },
    )


_TASKS: tuple[TaskSpec, ...] = (
    _mcp_task(
        "mcp-echo",
        difficulty="easy",
        prompt="Use the MCP echo tool to echo the message 'Hello MCP!'.",
        expected="Echo: Hello MCP!",
        required_tools=("echo",),
        expected_args={"message": "Hello MCP!"},
        expected_fragment="Echo: Hello MCP!",
        trap="Returning the message without the 'Echo: ' prefix, or answering without calling the echo tool.",
    ),
    _mcp_task(
        "mcp-sum-small",
        difficulty="easy",
        prompt="Use the MCP arithmetic tool (get-sum or add) to calculate the sum of 17 and 25.",
        expected="42",
        required_tools=("get-sum", "add"),
        expected_args={"a": 17, "b": 25},
        expected_fragment="42",
        trap="Performing arithmetic without calling the MCP tool, or using wrong argument names.",
    ),
    _mcp_task(
        "mcp-sum-negative",
        difficulty="medium",
        prompt="Use the MCP arithmetic tool (get-sum or add) to add -30 and 85.",
        expected="55",
        required_tools=("get-sum", "add"),
        expected_args={"a": -30, "b": 85},
        expected_fragment="55",
        trap="Treating negative numbers as positive (producing 115).",
    ),
    _mcp_task(
        "mcp-annotated-message",
        difficulty="medium",
        prompt="Use the MCP get-annotated-message tool with messageType 'debug' to inspect annotations.",
        expected="debug",
        required_tools=("get-annotated-message", "annotatedMessage"),
        expected_args={"messageType": "debug"},
        expected_fragment="debug",
        trap="Requesting 'info' or 'error' instead of 'debug'.",
    ),
    _mcp_task(
        "mcp-get-env",
        difficulty="hard",
        prompt="Use the MCP get-env (or printEnv) tool to inspect the environment variables and report the PATH variable.",
        expected="PATH",
        required_tools=("get-env", "printEnv"),
        expected_args={},
        expected_fragment="PATH",
        trap="Guessing environment variables rather than querying the live server.",
    ),
    _mcp_task(
        "mcp-tiny-image",
        difficulty="hard",
        prompt="Call the MCP get-tiny-image tool to retrieve the tiny MCP logo image.",
        expected="image",
        required_tools=("get-tiny-image",),
        expected_args={},
        expected_fragment="image",
        trap="Fabricating an image URL instead of calling get-tiny-image.",
    ),
)

_SUITE = DomainSuite(domain=DOMAIN, tasks=_TASKS)


def build_suite() -> DomainSuite:
    """Build the mcp_everything suite."""
    return _SUITE


def load_suite() -> DomainSuite:
    """Return the frozen mcp_everything task suite."""
    return _SUITE

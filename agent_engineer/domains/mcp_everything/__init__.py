"""The mcp_everything domain: real Model Context Protocol tool integration.

Exports the two required harness callables:
    suite = get_suite()          # -> DomainSuite
    evaluator = get_evaluator()  # -> TaskEvaluator

Also exports the ToolRuntime adapter:
    tool_runtime = get_tool_runtime() # -> MCPToolRuntime
"""

from __future__ import annotations

from agent_engineer.domains.mcp_everything.client import (
    MCPClient,
    MCPServerError,
    MCPServerNotFoundError,
    MCPTransport,
    StdioTransport,
    StubTransport,
)
from agent_engineer.domains.mcp_everything.evaluator import EVALUATOR_ID, evaluate
from agent_engineer.domains.mcp_everything.runtime import MCPToolRuntime
from agent_engineer.domains.mcp_everything.tasks import DOMAIN, build_suite, load_suite
from agent_engineer.evaluation import DomainSuite, TaskEvaluator

__all__ = [
    "DOMAIN",
    "EVALUATOR_ID",
    "get_suite",
    "get_evaluator",
    "get_tool_runtime",
    "MCPClient",
    "MCPToolRuntime",
    "MCPServerError",
    "MCPServerNotFoundError",
    "MCPTransport",
    "StdioTransport",
    "StubTransport",
]


def get_suite() -> DomainSuite:
    """Return the frozen mcp_everything task suite."""
    return load_suite()


def get_evaluator() -> TaskEvaluator:
    """Return the mcp_everything TaskEvaluator: (TaskSpec, Trajectory) -> EvaluatorVerdict."""
    return evaluate


def get_tool_runtime(client: MCPClient | None = None) -> MCPToolRuntime:
    """Return an MCPToolRuntime adapter instance."""
    return MCPToolRuntime(client=client)

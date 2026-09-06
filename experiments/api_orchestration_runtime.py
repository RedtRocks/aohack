"""ToolRuntime adapter for the api_orchestration domain, used by the real-model
gate experiments.

Modelled on ``agent_engineer.domains.mcp_everything.runtime.MCPToolRuntime``,
but kept out of ``agent_engineer/domains/api_orchestration`` on purpose: that
package's own test suite (``tests/test_api_orchestration.py::
test_this_domain_imports_no_engine_module``) forbids it from importing
``agent_engineer.ports`` at all -- unlike ``mcp_everything``, this domain's
architecture keeps every engine-execution concern on the caller's side of the
line. So the adapter lives here, in the caller, and reads only the metadata
fixture contract documented in ``agent_engineer/evaluation/INTERFACE.md``:
``task.metadata["tool_schemas"]`` (a tuple of dicts shaped like
:class:`ToolSchema`) and ``task.metadata["call_tool"]`` (a
``(tool_name, args) -> Any`` callable that raises on failure, built per task
in ``agent_engineer/domains/api_orchestration/tasks.py``).
"""

from __future__ import annotations

from typing import Any

from agent_engineer.evaluation import TaskSpec
from agent_engineer.ports import ToolResult, ToolRuntime, ToolSchema


class ApiOrchestrationToolRuntime:
    """Adapts api_orchestration's per-task ``call_tool``/``tool_schemas``
    metadata to the engine's ToolRuntime protocol.

    ``schemas()`` takes no task, so the catalogue is collected once, up
    front, from every task in the suite (every task in this domain exposes
    the same catalogue -- see ``tasks.py``'s ``TOOL_NAMES`` default).
    ``invoke`` is inherently per-task: it re-reads ``call_tool`` off the
    specific task on each call, since dispatch closes over that task's own
    simulated API instance.
    """

    def __init__(self, tasks: tuple[TaskSpec, ...]) -> None:
        by_name: dict[str, ToolSchema] = {}
        for task in tasks:
            for raw in task.metadata.get("tool_schemas", ()):
                schema = raw if isinstance(raw, ToolSchema) else ToolSchema(**raw)
                by_name[schema.name] = schema
        self._schemas = tuple(by_name.values())

    def schemas(self) -> tuple[ToolSchema, ...]:
        return self._schemas

    def invoke(self, task: TaskSpec, tool_name: str, args: dict[str, Any]) -> ToolResult:
        call_tool = task.metadata.get("call_tool")
        if call_tool is None:
            return ToolResult(error=f"task {task.task_id!r} carries no call_tool fixture")
        try:
            return ToolResult(value=call_tool(tool_name, args))
        except Exception as exc:
            return ToolResult(error=f"{type(exc).__name__}: {exc}")

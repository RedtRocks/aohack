"""The interfaces the engine consumes from the outside world.

Everything domain-specific enters the engine through this module and nowhere
else. The engine itself imports only the standard library, pydantic,
:mod:`agent_engineer.schemas`, and the measurement package; a domain reaches it
as an object satisfying one of the protocols below, handed in by the caller.

Two halves, with a firm line between them:

* **Measurement's, not ours.** ``TaskSpec``, ``DomainSuite`` and
  ``TaskEvaluator`` are defined in :mod:`agent_engineer.evaluation` and
  documented in ``agent_engineer/evaluation/INTERFACE.md``. They are re-exported
  here for convenience only. The engine does not define them, extend them, or
  grade anything itself.
* **Ours.** ``ToolSchema``, ``ToolResult``, ``ToolRuntime``, ``AgentAction``,
  ``ModelBackend`` and ``TextGenerator`` are agent-execution concerns. They
  describe how a candidate agent acts, which is the engine's business.

The engine supplies the harness exactly one callable, :class:`TaskRunner`:
``(spec, task, attempt) -> Trajectory``. It may raise; a raised run is the
harness's to record as a failure. The returned trajectory must carry
``tokens`` and ``elapsed_seconds``, since cost and speed are read from those
fields and nowhere else.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_engineer.schemas import AgentSpec, ToolCall, Trajectory

try:  # pragma: no cover - exercised by whichever half of the tree is present
    from agent_engineer.evaluation import DomainSuite, TaskEvaluator, TaskSpec
except ImportError:  # pragma: no cover
    # The measurement package has not landed on this branch yet. These stand-ins
    # exist so the engine can be developed and tested against the agreed shape;
    # they are deleted, not merged, the moment agent_engineer.evaluation is on
    # integration. Nothing here grades, aggregates, or counts anything.

    @runtime_checkable
    class TaskSpec(Protocol):  # type: ignore[no-redef]
        """One unit of work. ``metadata`` carries domain fixtures, opaque to the engine."""

        @property
        def task_id(self) -> str: ...

        @property
        def prompt(self) -> str: ...

        @property
        def metadata(self) -> dict[str, Any]: ...

    @runtime_checkable
    class DomainSuite(Protocol):  # type: ignore[no-redef]
        """A named, re-iterable set of tasks for one domain."""

        @property
        def suite_id(self) -> str: ...

        @property
        def tasks(self) -> tuple[TaskSpec, ...]: ...

    @runtime_checkable
    class TaskEvaluator(Protocol):  # type: ignore[no-redef]
        """Grades one finished trajectory. Grades only -- never counts or averages."""

        def __call__(self, task: TaskSpec, trajectory: Trajectory) -> Any: ...


__all__ = [
    "TaskSpec",
    "DomainSuite",
    "TaskEvaluator",
    "TaskRunner",
    "ToolSchema",
    "ToolResult",
    "ToolRuntime",
    "AgentAction",
    "ModelBackend",
    "TextGenerator",
]


@runtime_checkable
class TaskRunner(Protocol):
    """What the engine hands the harness: one spec, one task, one attempt, one trajectory.

    ``attempt`` distinguishes repeat runs of the same pair, so the harness can
    ask for several samples and the trajectory ids stay distinct.
    """

    def __call__(self, spec: AgentSpec, task: TaskSpec, attempt: int) -> Trajectory: ...


class ToolSchema(BaseModel):
    """A tool as advertised to the model: name, description, JSON-schema parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolResult(BaseModel):
    """What a tool returned. ``error`` set means the call failed and ``value`` is ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class ToolRuntime(Protocol):
    """The tools available in a domain, and the ability to actually call one."""

    def schemas(self) -> tuple[ToolSchema, ...]: ...

    def invoke(self, task: TaskSpec, tool_name: str, args: dict[str, Any]) -> ToolResult: ...


class AgentAction(BaseModel):
    """One decision by the model: call a tool, or finish with an answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    thought: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_final(self) -> bool:
        return self.tool_name is None


@runtime_checkable
class ModelBackend(Protocol):
    """Decides the next action given the spec, the task, and what has happened so far.

    The backend receives the spec's system prompt, tool list and strategy and is
    expected to honour them; it is the only component that needs a model.
    """

    def next_action(
        self,
        spec: AgentSpec,
        task: TaskSpec,
        tools: tuple[ToolSchema, ...],
        history: tuple[ToolCall, ...],
    ) -> AgentAction: ...


@runtime_checkable
class TextGenerator(Protocol):
    """Free-form text completion, used by the model-assisted stage implementations.

    Kept separate from :class:`ModelBackend` so the loop's own stages can use a
    model while the agent under test runs on something else, or on neither.
    """

    def complete(self, system: str, prompt: str) -> str: ...

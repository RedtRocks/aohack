"""The interfaces the engine consumes from the outside world.

Everything domain-specific enters the engine through this module and nowhere
else. The engine itself imports only the standard library, pydantic, and
:mod:`agent_engineer`; a domain reaches it as an object satisfying one of the
protocols below, handed in by the caller.

Four ports:

* :class:`Task` / :class:`TaskSet` -- what the agent is asked to do.
* :class:`Evaluator` -- whether it did it. Owned by the measurement side.
* :class:`ToolRuntime` -- the tools it may call, and what happens when it does.
* :class:`ModelBackend` -- what decides the next action.

The protocols are structural and ``runtime_checkable``, so a domain package
satisfies them by shape alone: nothing on the domain side needs to import the
engine, and the engine never needs to name a domain.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_engineer.schemas import AgentSpec, EvaluatorVerdict, ToolCall


@runtime_checkable
class Task(Protocol):
    """One unit of work handed to an agent.

    ``prompt`` is the only text the engine reads, and it is passed through
    verbatim -- the engine never inspects or rewrites it, so the domain keeps
    full control of what its tasks say.
    """

    @property
    def task_id(self) -> str: ...

    @property
    def prompt(self) -> str: ...


@runtime_checkable
class TaskSet(Protocol):
    """A named, ordered, re-iterable collection of :class:`Task`.

    Held-out splits are the domain's business: the engine evaluates whatever
    set it is given and compares like against like across generations.
    """

    @property
    def task_set_id(self) -> str: ...

    def tasks(self) -> tuple[Task, ...]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Judges one finished run. Owned by the measurement side.

    The engine calls this once per trajectory and stores the verdict verbatim;
    it never second-guesses a score or re-derives ``passed``.
    """

    @property
    def evaluator_id(self) -> str: ...

    def evaluate(self, task: Task, final_answer: str | None, tool_calls: tuple[ToolCall, ...]) -> EvaluatorVerdict: ...


class ToolSchema(BaseModel):
    """A tool as advertised to the model: name, description, JSON-schema parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolResult(BaseModel):
    """What a tool returned. Exactly one of ``value`` or ``error`` is meaningful."""

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

    def invoke(self, task: Task, tool_name: str, args: dict[str, Any]) -> ToolResult: ...


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

    The backend receives the spec's system prompt and strategy and is expected
    to honour them; it is the only component that needs to talk to a model.
    """

    def next_action(
        self,
        spec: AgentSpec,
        task: Task,
        tools: tuple[ToolSchema, ...],
        history: tuple[ToolCall, ...],
    ) -> AgentAction: ...


@runtime_checkable
class TextGenerator(Protocol):
    """Free-form text completion, used by the model-assisted stage implementations.

    Kept separate from :class:`ModelBackend` so the loop can run its stages with
    a real model while the agent under test runs on something else, or vice
    versa, or with neither.
    """

    def complete(self, system: str, prompt: str) -> str: ...

"""The four frozen typed schemas shared by every stage of the automated-agent-engineer build.

This module owns data only. It contains no stage logic, no engine, no domain code.

The four schemas are:

* :class:`AgentSpec` -- the candidate agent under optimization.
* :class:`Trajectory` -- one recorded run of an ``AgentSpec`` against one task.
* :class:`Diagnosis` -- why a set of trajectories failed, using a closed cause enum.
* :class:`Mutation` -- a single edit to an ``AgentSpec``, and the Diagnosis that motivated it.

Every schema is an immutable dataclass, validates itself on construction, and
round-trips through ``to_dict``/``from_dict`` and ``to_json``/``from_json``.
Serialization is deterministic: the same object always produces byte-identical
JSON, so specs can be stored, hashed, and diffed as text.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SchemaError",
    "SCHEMA_VERSION",
    "OrchestrationStrategy",
    "MemoryKind",
    "MemoryConfig",
    "StoppingConditions",
    "AgentSpec",
    "ToolCall",
    "TokenUsage",
    "EvaluatorVerdict",
    "Trajectory",
    "FailureCause",
    "FailureAttribution",
    "Diagnosis",
    "MutationKind",
    "Mutation",
    "diff_agent_specs",
]

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a schema object is constructed or decoded with invalid data."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _require_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field_name} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise SchemaError(f"{field_name} must be non-empty")
    return value


def _require_non_negative(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field_name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise SchemaError(f"{field_name} must be >= 0, got {value}")
    return value


def _require_seconds(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise SchemaError(f"{field_name} must be >= 0, got {value}")
    return float(value)


def _require_unit_interval(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field_name} must be a number, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise SchemaError(f"{field_name} must be within [0.0, 1.0], got {value}")
    return float(value)


def _optional_positive(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    count = _require_non_negative(value, field_name)
    if count == 0:
        raise SchemaError(f"{field_name} must be > 0 when set")
    return count


class _FrozenMapping(Mapping):
    """A hashable, key-sorted read-only mapping used for tool-call payloads."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[tuple[str, Any]]) -> None:
        self._items: tuple[tuple[str, Any], ...] = tuple(sorted(items, key=lambda pair: pair[0]))

    def __getitem__(self, key: str) -> Any:
        for existing, value in self._items:
            if existing == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenMapping):
            return self._items == other._items
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"_FrozenMapping({dict(self)!r})"


def _freeze_json(value: Any, field_name: str) -> Any:
    """Recursively convert a JSON-compatible value into an immutable equivalent.

    Mappings become :class:`_FrozenMapping`, sequences become tuples. Anything
    that is not JSON-representable is rejected here rather than at
    serialization time, so an invalid payload fails at construction.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{field_name} keys must be strings, got {type(key).__name__}")
            items.append((key, _freeze_json(item, f"{field_name}[{key!r}]")))
        return _FrozenMapping(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{field_name}[]") for item in value)
    raise SchemaError(f"{field_name} must be JSON-representable, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    """Inverse of :func:`_freeze_json`: back to plain dicts and lists."""
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _decode_enum(enum_cls, value: Any, field_name: str):
    if isinstance(value, enum_cls):
        return value
    _require_text(value, field_name)
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(sorted(member.value for member in enum_cls))
        raise SchemaError(f"{field_name} must be one of: {allowed}; got {value!r}") from None


def _require_mapping(payload: Any, schema_name: str) -> Mapping:
    if not isinstance(payload, Mapping):
        raise SchemaError(f"{schema_name} payload must be a mapping, got {type(payload).__name__}")
    return payload


def _pull(payload: Mapping, key: str, schema_name: str) -> Any:
    if key not in payload:
        raise SchemaError(f"{schema_name} payload is missing required key {key!r}")
    return payload[key]


def _dumps(payload: Mapping, *, indent: int | None) -> str:
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)


def _normalize_tool_names(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise SchemaError(f"{field_name} must be a sequence of tool names")
    names: list[str] = []
    for value in values:
        name = _require_text(value, f"{field_name}[]")
        if name in names:
            raise SchemaError(f"{field_name} contains duplicate tool name {name!r}")
        names.append(name)
    return tuple(names)


# --------------------------------------------------------------------------- #
# 1. AgentSpec
# --------------------------------------------------------------------------- #


class OrchestrationStrategy(Enum):
    """How the agent loop drives the model. Closed set."""

    SINGLE_SHOT = "single_shot"
    REACT = "react"
    PLAN_THEN_EXECUTE = "plan_then_execute"
    REFLEXION = "reflexion"
    TREE_SEARCH = "tree_search"
    DELEGATING_SUBAGENTS = "delegating_subagents"


class MemoryKind(Enum):
    """How context is carried across steps. Closed set."""

    NONE = "none"
    FULL_TRANSCRIPT = "full_transcript"
    SCRATCHPAD = "scratchpad"
    ROLLING_SUMMARY = "rolling_summary"
    VECTOR_RETRIEVAL = "vector_retrieval"
    EPISODIC_STORE = "episodic_store"


_RETRIEVAL_KINDS = (MemoryKind.VECTOR_RETRIEVAL, MemoryKind.EPISODIC_STORE)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Memory configuration for an :class:`AgentSpec`."""

    kind: MemoryKind = MemoryKind.FULL_TRANSCRIPT
    max_tokens: int | None = None
    """Token budget for retained memory. ``None`` means unbounded."""
    retrieval_k: int | None = None
    """Items fetched per retrieval. Required for, and only for, retrieval kinds."""
    persist_across_runs: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MemoryKind):
            raise SchemaError("MemoryConfig.kind must be a MemoryKind")
        object.__setattr__(
            self, "max_tokens", _optional_positive(self.max_tokens, "MemoryConfig.max_tokens")
        )
        object.__setattr__(
            self, "retrieval_k", _optional_positive(self.retrieval_k, "MemoryConfig.retrieval_k")
        )
        if not isinstance(self.persist_across_runs, bool):
            raise SchemaError("MemoryConfig.persist_across_runs must be a bool")
        if self.kind in _RETRIEVAL_KINDS and self.retrieval_k is None:
            raise SchemaError(f"MemoryConfig.retrieval_k is required for kind {self.kind.value}")
        if self.kind not in _RETRIEVAL_KINDS and self.retrieval_k is not None:
            raise SchemaError(
                f"MemoryConfig.retrieval_k is not meaningful for kind {self.kind.value}"
            )
        if self.kind is MemoryKind.NONE and self.max_tokens is not None:
            raise SchemaError("MemoryConfig.max_tokens is not meaningful for kind none")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "max_tokens": self.max_tokens,
            "retrieval_k": self.retrieval_k,
            "persist_across_runs": self.persist_across_runs,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "MemoryConfig":
        data = _require_mapping(payload, "MemoryConfig")
        return cls(
            kind=_decode_enum(MemoryKind, _pull(data, "kind", "MemoryConfig"), "MemoryConfig.kind"),
            max_tokens=data.get("max_tokens"),
            retrieval_k=data.get("retrieval_k"),
            persist_across_runs=bool(data.get("persist_across_runs", False)),
        )


@dataclass(frozen=True, slots=True)
class StoppingConditions:
    """When the agent loop must halt. At least one bound is required."""

    max_steps: int | None = None
    max_tokens: int | None = None
    max_wall_clock_seconds: float | None = None
    stop_on_tools: tuple[str, ...] = ()
    """Tool names that terminate the run as soon as they are called."""
    require_final_answer: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_steps", _optional_positive(self.max_steps, "StoppingConditions.max_steps")
        )
        object.__setattr__(
            self, "max_tokens", _optional_positive(self.max_tokens, "StoppingConditions.max_tokens")
        )
        if self.max_wall_clock_seconds is not None:
            seconds = _require_seconds(
                self.max_wall_clock_seconds, "StoppingConditions.max_wall_clock_seconds"
            )
            if seconds == 0.0:
                raise SchemaError(
                    "StoppingConditions.max_wall_clock_seconds must be > 0 when set"
                )
            object.__setattr__(self, "max_wall_clock_seconds", seconds)
        object.__setattr__(
            self,
            "stop_on_tools",
            _normalize_tool_names(self.stop_on_tools, "StoppingConditions.stop_on_tools"),
        )
        if not isinstance(self.require_final_answer, bool):
            raise SchemaError("StoppingConditions.require_final_answer must be a bool")
        bounds = (self.max_steps, self.max_tokens, self.max_wall_clock_seconds)
        if all(bound is None for bound in bounds) and not self.stop_on_tools:
            raise SchemaError(
                "StoppingConditions needs at least one of max_steps, max_tokens, "
                "max_wall_clock_seconds, or stop_on_tools"
            )

    def to_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "stop_on_tools": list(self.stop_on_tools),
            "require_final_answer": self.require_final_answer,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "StoppingConditions":
        data = _require_mapping(payload, "StoppingConditions")
        return cls(
            max_steps=data.get("max_steps"),
            max_tokens=data.get("max_tokens"),
            max_wall_clock_seconds=data.get("max_wall_clock_seconds"),
            stop_on_tools=tuple(data.get("stop_on_tools") or ()),
            require_final_answer=bool(data.get("require_final_answer", True)),
        )


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """A candidate agent configuration: the unit the engineer loop mutates.

    Two specs describing the same agent compare equal and render to the same
    canonical text, so ``diff_agent_specs`` between them is empty.
    """

    spec_id: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    """Names of the tools exposed to the agent. Order is preserved and diffed."""
    strategy: OrchestrationStrategy = OrchestrationStrategy.REACT
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    stopping: StoppingConditions = field(default_factory=lambda: StoppingConditions(max_steps=20))
    parent_spec_id: str | None = None
    """The spec this one was mutated from, if any."""

    def __post_init__(self) -> None:
        _require_text(self.spec_id, "AgentSpec.spec_id")
        _require_text(self.system_prompt, "AgentSpec.system_prompt")
        object.__setattr__(self, "tools", _normalize_tool_names(self.tools, "AgentSpec.tools"))
        if not isinstance(self.strategy, OrchestrationStrategy):
            raise SchemaError("AgentSpec.strategy must be an OrchestrationStrategy")
        if not isinstance(self.memory, MemoryConfig):
            raise SchemaError("AgentSpec.memory must be a MemoryConfig")
        if not isinstance(self.stopping, StoppingConditions):
            raise SchemaError("AgentSpec.stopping must be a StoppingConditions")
        if self.parent_spec_id is not None:
            _require_text(self.parent_spec_id, "AgentSpec.parent_spec_id")
            if self.parent_spec_id == self.spec_id:
                raise SchemaError("AgentSpec.parent_spec_id must differ from spec_id")
        unknown = [name for name in self.stopping.stop_on_tools if name not in self.tools]
        if unknown:
            raise SchemaError(
                "StoppingConditions.stop_on_tools references tools not exposed by the spec: "
                + ", ".join(sorted(unknown))
            )

    def to_dict(self) -> dict:
        return {
            "schema": "AgentSpec",
            "schema_version": SCHEMA_VERSION,
            "spec_id": self.spec_id,
            "system_prompt": self.system_prompt,
            "tools": list(self.tools),
            "strategy": self.strategy.value,
            "memory": self.memory.to_dict(),
            "stopping": self.stopping.to_dict(),
            "parent_spec_id": self.parent_spec_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AgentSpec":
        data = _require_mapping(payload, "AgentSpec")
        return cls(
            spec_id=_pull(data, "spec_id", "AgentSpec"),
            system_prompt=_pull(data, "system_prompt", "AgentSpec"),
            tools=tuple(data.get("tools") or ()),
            strategy=_decode_enum(
                OrchestrationStrategy, _pull(data, "strategy", "AgentSpec"), "AgentSpec.strategy"
            ),
            memory=MemoryConfig.from_dict(_pull(data, "memory", "AgentSpec")),
            stopping=StoppingConditions.from_dict(_pull(data, "stopping", "AgentSpec")),
            parent_spec_id=data.get("parent_spec_id"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return _dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "AgentSpec":
        return cls.from_dict(json.loads(text))

    def with_changes(self, **changes: Any) -> "AgentSpec":
        """Return a validated copy with the given fields replaced."""
        return replace(self, **changes)

    def to_text(self) -> str:
        """Render the spec as canonical, line-oriented, diff-friendly text.

        One fact per line, so a change to one field produces one changed line.
        The system prompt is emitted last, indented, one output line per prompt
        line, so prompt edits diff at line granularity.
        """
        memory_bits = [f"kind={self.memory.kind.value}"]
        if self.memory.max_tokens is not None:
            memory_bits.append(f"max_tokens={self.memory.max_tokens}")
        if self.memory.retrieval_k is not None:
            memory_bits.append(f"retrieval_k={self.memory.retrieval_k}")
        memory_bits.append(f"persist_across_runs={str(self.memory.persist_across_runs).lower()}")

        stopping = self.stopping
        lines = [
            f"spec_id: {self.spec_id}",
            f"parent_spec_id: {self.parent_spec_id or '-'}",
            f"strategy: {self.strategy.value}",
            f"memory: {', '.join(memory_bits)}",
            "tools:",
        ]
        if self.tools:
            lines.extend(f"  - {name}" for name in self.tools)
        else:
            lines.append("  (none)")
        lines.append("stopping:")
        lines.append(f"  max_steps: {_render_optional(stopping.max_steps)}")
        lines.append(f"  max_tokens: {_render_optional(stopping.max_tokens)}")
        lines.append(
            f"  max_wall_clock_seconds: {_render_optional(stopping.max_wall_clock_seconds)}"
        )
        lines.append(f"  stop_on_tools: {', '.join(stopping.stop_on_tools) or '-'}")
        lines.append(f"  require_final_answer: {str(stopping.require_final_answer).lower()}")
        lines.append("system_prompt: |")
        lines.extend(f"  {line}" for line in self.system_prompt.splitlines() or [""])
        return "\n".join(lines) + "\n"


def _render_optional(value: Any) -> str:
    return "-" if value is None else str(value)


def diff_agent_specs(before: AgentSpec, after: AgentSpec, *, context_lines: int = 3) -> str:
    """Return a human-readable unified diff between two specs.

    Empty string when the two specs are equivalent. Each side is its canonical
    :meth:`AgentSpec.to_text` rendering, so a prompt edit shows as changed
    prompt lines and a tool change as an added or removed tool line.
    """
    if not isinstance(before, AgentSpec) or not isinstance(after, AgentSpec):
        raise SchemaError("diff_agent_specs requires two AgentSpec instances")
    diff = difflib.unified_diff(
        before.to_text().splitlines(keepends=True),
        after.to_text().splitlines(keepends=True),
        fromfile=f"a/{before.spec_id}",
        tofile=f"b/{after.spec_id}",
        n=context_lines,
    )
    return "".join(diff)


# --------------------------------------------------------------------------- #
# 2. Trajectory
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a run or a single call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        _require_non_negative(self.prompt_tokens, "TokenUsage.prompt_tokens")
        _require_non_negative(self.completion_tokens, "TokenUsage.completion_tokens")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TokenUsage":
        data = _require_mapping(payload, "TokenUsage")
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation inside a trajectory, with its arguments and result."""

    ordinal: int
    """0-based position in the trajectory. Must be contiguous and ascending."""
    tool_name: str
    args: Mapping = field(default_factory=dict)
    result: Any = None
    """The tool's return value on success; ``None`` when ``error`` is set."""
    error: str | None = None
    """Failure message, or ``None`` if the call succeeded."""
    elapsed_seconds: float = 0.0
    tokens: TokenUsage | None = None
    """Per-call token usage when the harness records it."""

    def __post_init__(self) -> None:
        _require_non_negative(self.ordinal, "ToolCall.ordinal")
        _require_text(self.tool_name, "ToolCall.tool_name")
        object.__setattr__(
            self, "args", _freeze_json(dict(_require_mapping(self.args, "ToolCall.args")), "ToolCall.args")
        )
        object.__setattr__(self, "result", _freeze_json(self.result, "ToolCall.result"))
        if self.error is not None:
            _require_text(self.error, "ToolCall.error")
            if self.result is not None:
                raise SchemaError("ToolCall cannot carry both a result and an error")
        object.__setattr__(
            self, "elapsed_seconds", _require_seconds(self.elapsed_seconds, "ToolCall.elapsed_seconds")
        )
        if self.tokens is not None and not isinstance(self.tokens, TokenUsage):
            raise SchemaError("ToolCall.tokens must be a TokenUsage or None")

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "ordinal": self.ordinal,
            "tool_name": self.tool_name,
            "args": _thaw_json(self.args),
            "result": _thaw_json(self.result),
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens": self.tokens.to_dict() if self.tokens is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ToolCall":
        data = _require_mapping(payload, "ToolCall")
        raw_tokens = data.get("tokens")
        return cls(
            ordinal=_pull(data, "ordinal", "ToolCall"),
            tool_name=_pull(data, "tool_name", "ToolCall"),
            args=data.get("args") or {},
            result=data.get("result"),
            error=data.get("error"),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            tokens=TokenUsage.from_dict(raw_tokens) if raw_tokens is not None else None,
        )


@dataclass(frozen=True, slots=True)
class EvaluatorVerdict:
    """The evaluator's judgement of a trajectory's final answer."""

    evaluator_id: str
    passed: bool
    score: float = 0.0
    """Normalized to [0.0, 1.0]."""
    rationale: str = ""

    def __post_init__(self) -> None:
        _require_text(self.evaluator_id, "EvaluatorVerdict.evaluator_id")
        if not isinstance(self.passed, bool):
            raise SchemaError("EvaluatorVerdict.passed must be a bool")
        object.__setattr__(self, "score", _require_unit_interval(self.score, "EvaluatorVerdict.score"))
        _require_text(self.rationale, "EvaluatorVerdict.rationale", allow_empty=True)

    def to_dict(self) -> dict:
        return {
            "evaluator_id": self.evaluator_id,
            "passed": self.passed,
            "score": self.score,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "EvaluatorVerdict":
        data = _require_mapping(payload, "EvaluatorVerdict")
        return cls(
            evaluator_id=_pull(data, "evaluator_id", "EvaluatorVerdict"),
            passed=bool(_pull(data, "passed", "EvaluatorVerdict")),
            score=data.get("score", 0.0),
            rationale=data.get("rationale", ""),
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One recorded run of an :class:`AgentSpec` against one task."""

    trajectory_id: str
    spec_id: str
    task_id: str
    tool_calls: tuple[ToolCall, ...] = ()
    """Ordered by ``ToolCall.ordinal``, contiguous from 0."""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0
    """Wall-clock duration of the whole run, including model latency."""
    final_answer: str | None = None
    """``None`` when the run stopped without producing an answer."""
    verdict: EvaluatorVerdict | None = None
    """``None`` when the trajectory has not been evaluated yet."""

    def __post_init__(self) -> None:
        _require_text(self.trajectory_id, "Trajectory.trajectory_id")
        _require_text(self.spec_id, "Trajectory.spec_id")
        _require_text(self.task_id, "Trajectory.task_id")
        calls = tuple(self.tool_calls)
        for index, call in enumerate(calls):
            if not isinstance(call, ToolCall):
                raise SchemaError("Trajectory.tool_calls must contain ToolCall instances")
            if call.ordinal != index:
                raise SchemaError(
                    "Trajectory.tool_calls must be contiguous and ascending from 0; "
                    f"position {index} has ordinal {call.ordinal}"
                )
        object.__setattr__(self, "tool_calls", calls)
        if not isinstance(self.tokens, TokenUsage):
            raise SchemaError("Trajectory.tokens must be a TokenUsage")
        object.__setattr__(
            self, "elapsed_seconds", _require_seconds(self.elapsed_seconds, "Trajectory.elapsed_seconds")
        )
        if self.final_answer is not None:
            _require_text(self.final_answer, "Trajectory.final_answer", allow_empty=True)
        if self.verdict is not None and not isinstance(self.verdict, EvaluatorVerdict):
            raise SchemaError("Trajectory.verdict must be an EvaluatorVerdict or None")

    @property
    def step_count(self) -> int:
        return len(self.tool_calls)

    @property
    def failed(self) -> bool:
        """True when an evaluated trajectory did not pass. Unevaluated runs are not failures."""
        return self.verdict is not None and not self.verdict.passed

    def to_dict(self) -> dict:
        return {
            "schema": "Trajectory",
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": self.trajectory_id,
            "spec_id": self.spec_id,
            "task_id": self.task_id,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tokens": self.tokens.to_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "final_answer": self.final_answer,
            "verdict": self.verdict.to_dict() if self.verdict is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Trajectory":
        data = _require_mapping(payload, "Trajectory")
        raw_verdict = data.get("verdict")
        return cls(
            trajectory_id=_pull(data, "trajectory_id", "Trajectory"),
            spec_id=_pull(data, "spec_id", "Trajectory"),
            task_id=_pull(data, "task_id", "Trajectory"),
            tool_calls=tuple(ToolCall.from_dict(item) for item in data.get("tool_calls") or ()),
            tokens=TokenUsage.from_dict(data.get("tokens") or {}),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
            final_answer=data.get("final_answer"),
            verdict=EvaluatorVerdict.from_dict(raw_verdict) if raw_verdict is not None else None,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return _dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Trajectory":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# 3. Diagnosis
# --------------------------------------------------------------------------- #


class FailureCause(Enum):
    """The closed set of causes a failure may be attributed to.

    Closed on purpose: a diagnosis stage must map every failure into one of
    these buckets so that mutations can be selected from a finite, testable
    table. Add a member here only alongside a mutation that can act on it.
    """

    MISSING_TOOL = "missing_tool"
    """The task needed a capability no exposed tool provides."""
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    """A suitable tool existed; the agent chose a different one."""
    TOOL_CALLED_WITH_BAD_ARGS = "tool_called_with_bad_args"
    TOOL_ERROR_UNHANDLED = "tool_error_unhandled"
    """A tool returned an error and the agent did not recover."""
    PROMPT_AMBIGUOUS = "prompt_ambiguous"
    """The system prompt underspecified what to do."""
    INSTRUCTION_IGNORED = "instruction_ignored"
    """The system prompt said it; the agent did not follow it."""
    PLANNING_FAILURE = "planning_failure"
    """Steps were individually fine but sequenced or decomposed wrongly."""
    REPEATED_LOOP = "repeated_loop"
    CONTEXT_OVERFLOW = "context_overflow"
    MEMORY_LOSS = "memory_loss"
    """Information was gathered and then dropped before it was needed."""
    PREMATURE_STOP = "premature_stop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    """Hit a stopping condition before finishing."""
    HALLUCINATED_FACT = "hallucinated_fact"
    OUTPUT_FORMAT_VIOLATION = "output_format_violation"
    EVALUATOR_ERROR = "evaluator_error"
    """The agent was right; the verdict was wrong."""
    UNKNOWN = "unknown"
    """Explicitly undiagnosed. Never inferred silently -- it must be chosen."""


@dataclass(frozen=True, slots=True)
class FailureAttribution:
    """One failing trajectory mapped to exactly one cause."""

    trajectory_id: str
    cause: FailureCause
    evidence: str
    """Why this cause, citing what happened in the trajectory."""
    confidence: float = 1.0
    """Normalized to [0.0, 1.0]. Weights the dominant-cause vote."""
    step_ordinal: int | None = None
    """The tool call where the failure originated, when it is localizable."""

    def __post_init__(self) -> None:
        _require_text(self.trajectory_id, "FailureAttribution.trajectory_id")
        if not isinstance(self.cause, FailureCause):
            raise SchemaError("FailureAttribution.cause must be a FailureCause")
        _require_text(self.evidence, "FailureAttribution.evidence")
        object.__setattr__(
            self, "confidence", _require_unit_interval(self.confidence, "FailureAttribution.confidence")
        )
        if self.step_ordinal is not None:
            _require_non_negative(self.step_ordinal, "FailureAttribution.step_ordinal")

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "cause": self.cause.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "step_ordinal": self.step_ordinal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FailureAttribution":
        data = _require_mapping(payload, "FailureAttribution")
        return cls(
            trajectory_id=_pull(data, "trajectory_id", "FailureAttribution"),
            cause=_decode_enum(
                FailureCause, _pull(data, "cause", "FailureAttribution"), "FailureAttribution.cause"
            ),
            evidence=_pull(data, "evidence", "FailureAttribution"),
            confidence=data.get("confidence", 1.0),
            step_ordinal=data.get("step_ordinal"),
        )


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Per-failure causes for one spec, plus the aggregate dominant cause.

    ``dominant_cause`` is derived, not asserted: it is the cause with the
    greatest summed confidence across attributions, breaking ties by
    attribution count and then by declaration order in :class:`FailureCause`.
    It is ``None`` only when there are no attributions.
    """

    diagnosis_id: str
    spec_id: str
    attributions: tuple[FailureAttribution, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        _require_text(self.diagnosis_id, "Diagnosis.diagnosis_id")
        _require_text(self.spec_id, "Diagnosis.spec_id")
        attributions = tuple(self.attributions)
        seen: set[str] = set()
        for attribution in attributions:
            if not isinstance(attribution, FailureAttribution):
                raise SchemaError("Diagnosis.attributions must contain FailureAttribution instances")
            if attribution.trajectory_id in seen:
                raise SchemaError(
                    "Diagnosis has more than one attribution for trajectory "
                    f"{attribution.trajectory_id!r}; each failure gets exactly one cause"
                )
            seen.add(attribution.trajectory_id)
        object.__setattr__(self, "attributions", attributions)
        _require_text(self.notes, "Diagnosis.notes", allow_empty=True)

    @property
    def dominant_cause(self) -> FailureCause | None:
        if not self.attributions:
            return None
        order = {cause: index for index, cause in enumerate(FailureCause)}
        weight: Counter = Counter()
        count: Counter = Counter()
        for attribution in self.attributions:
            weight[attribution.cause] += attribution.confidence
            count[attribution.cause] += 1
        return min(weight, key=lambda cause: (-weight[cause], -count[cause], order[cause]))

    def cause_histogram(self) -> dict:
        """Attribution counts per cause, in :class:`FailureCause` declaration order."""
        counts = Counter(attribution.cause for attribution in self.attributions)
        return {cause: counts[cause] for cause in FailureCause if counts[cause]}

    def to_dict(self) -> dict:
        dominant = self.dominant_cause
        return {
            "schema": "Diagnosis",
            "schema_version": SCHEMA_VERSION,
            "diagnosis_id": self.diagnosis_id,
            "spec_id": self.spec_id,
            "attributions": [attribution.to_dict() for attribution in self.attributions],
            "dominant_cause": dominant.value if dominant is not None else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Diagnosis":
        data = _require_mapping(payload, "Diagnosis")
        diagnosis = cls(
            diagnosis_id=_pull(data, "diagnosis_id", "Diagnosis"),
            spec_id=_pull(data, "spec_id", "Diagnosis"),
            attributions=tuple(
                FailureAttribution.from_dict(item) for item in data.get("attributions") or ()
            ),
            notes=data.get("notes", ""),
        )
        # dominant_cause is derived; a payload that carries one must agree with it.
        encoded = data.get("dominant_cause")
        if encoded is not None:
            expected = _decode_enum(FailureCause, encoded, "Diagnosis.dominant_cause")
            actual = diagnosis.dominant_cause
            if expected is not actual:
                raise SchemaError(
                    f"Diagnosis.dominant_cause {encoded!r} disagrees with the attributions, "
                    f"which imply {actual.value if actual is not None else None!r}"
                )
        return diagnosis

    def to_json(self, *, indent: int | None = 2) -> str:
        return _dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Diagnosis":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# 4. Mutation
# --------------------------------------------------------------------------- #


class MutationKind(Enum):
    """The closed set of edits that may be applied to an :class:`AgentSpec`."""

    SYSTEM_PROMPT_REWRITE = "system_prompt_rewrite"
    TOOL_ADDED = "tool_added"
    TOOL_REMOVED = "tool_removed"
    TOOL_SET_REORDERED = "tool_set_reordered"
    STRATEGY_CHANGED = "strategy_changed"
    MEMORY_RECONFIGURED = "memory_reconfigured"
    STOPPING_ADJUSTED = "stopping_adjusted"


@dataclass(frozen=True, slots=True)
class Mutation:
    """One edit to an :class:`AgentSpec`, and the :class:`Diagnosis` that motivated it.

    A mutation always names the diagnosis and the specific cause it addresses,
    so an unmotivated edit cannot be recorded.
    """

    mutation_id: str
    kind: MutationKind
    target_path: str
    """Dotted field path into AgentSpec, e.g. ``system_prompt`` or ``memory.kind``."""
    before: str
    """Prior value rendered as text. Empty for a pure addition."""
    after: str
    """New value rendered as text. Empty for a pure removal."""
    rationale: str
    """Why this edit should fix the motivating cause."""
    motivating_diagnosis_id: str
    motivating_cause: FailureCause
    parent_spec_id: str
    child_spec_id: str | None = None
    """The spec produced by applying this mutation, once it exists."""

    def __post_init__(self) -> None:
        _require_text(self.mutation_id, "Mutation.mutation_id")
        if not isinstance(self.kind, MutationKind):
            raise SchemaError("Mutation.kind must be a MutationKind")
        _require_text(self.target_path, "Mutation.target_path")
        _require_text(self.before, "Mutation.before", allow_empty=True)
        _require_text(self.after, "Mutation.after", allow_empty=True)
        if self.before == self.after:
            raise SchemaError("Mutation.before and Mutation.after must differ")
        _require_text(self.rationale, "Mutation.rationale")
        _require_text(self.motivating_diagnosis_id, "Mutation.motivating_diagnosis_id")
        if not isinstance(self.motivating_cause, FailureCause):
            raise SchemaError("Mutation.motivating_cause must be a FailureCause")
        _require_text(self.parent_spec_id, "Mutation.parent_spec_id")
        if self.child_spec_id is not None:
            _require_text(self.child_spec_id, "Mutation.child_spec_id")
            if self.child_spec_id == self.parent_spec_id:
                raise SchemaError("Mutation.child_spec_id must differ from parent_spec_id")

    def to_dict(self) -> dict:
        return {
            "schema": "Mutation",
            "schema_version": SCHEMA_VERSION,
            "mutation_id": self.mutation_id,
            "kind": self.kind.value,
            "target_path": self.target_path,
            "before": self.before,
            "after": self.after,
            "rationale": self.rationale,
            "motivating_diagnosis_id": self.motivating_diagnosis_id,
            "motivating_cause": self.motivating_cause.value,
            "parent_spec_id": self.parent_spec_id,
            "child_spec_id": self.child_spec_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Mutation":
        data = _require_mapping(payload, "Mutation")
        return cls(
            mutation_id=_pull(data, "mutation_id", "Mutation"),
            kind=_decode_enum(MutationKind, _pull(data, "kind", "Mutation"), "Mutation.kind"),
            target_path=_pull(data, "target_path", "Mutation"),
            before=data.get("before", ""),
            after=data.get("after", ""),
            rationale=_pull(data, "rationale", "Mutation"),
            motivating_diagnosis_id=_pull(data, "motivating_diagnosis_id", "Mutation"),
            motivating_cause=_decode_enum(
                FailureCause, _pull(data, "motivating_cause", "Mutation"), "Mutation.motivating_cause"
            ),
            parent_spec_id=_pull(data, "parent_spec_id", "Mutation"),
            child_spec_id=data.get("child_spec_id"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return _dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Mutation":
        return cls.from_dict(json.loads(text))

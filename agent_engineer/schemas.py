"""The four frozen typed schemas shared by every stage of the automated-agent-engineer build.

This module owns data only. It contains no stage logic, no engine code, and
nothing domain-specific: every field here is meaningful for an agent working in
any domain whatsoever.

The four schemas are:

* :class:`AgentSpec` -- the candidate agent under optimization.
* :class:`Trajectory` -- one recorded run of an ``AgentSpec`` against one task.
* :class:`Diagnosis` -- why a set of trajectories failed, over a closed cause enum.
* :class:`Mutation` -- one edit to an ``AgentSpec``, carrying the Diagnosis that motivated it.

All models are pydantic v2, frozen, and reject unknown fields. JSON is the wire
format (``model_dump_json`` / ``model_validate_json``) and is deterministic, so
records are byte-stable and hashable. :class:`AgentSpec` additionally renders to
a stable line-oriented text form via :meth:`AgentSpec.to_text`, which is what
:func:`diff_agent_specs` compares -- so a ``git diff`` of two specs is legible
to a human.
"""

from __future__ import annotations

import difflib
from collections import Counter
from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)

__all__ = [
    "SCHEMA_VERSION",
    "OrchestrationStrategy",
    "MemoryKind",
    "MemoryConfig",
    "StoppingConditions",
    "AgentSpec",
    "TokenUsage",
    "ToolCall",
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

NonEmptyStr = Annotated[str, Field(min_length=1)]
"""A string that must contain at least one character."""

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
"""A score normalized to [0.0, 1.0]."""

ToolName = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
"""A tool identifier: non-empty, no leading or trailing whitespace."""

JsonValue = Any
"""A JSON-representable payload. Validated as JSON-serializable on dump."""


class _Frozen(BaseModel):
    """Base config for every schema in this module.

    Frozen, hashable, and strict about unknown fields. Derived values such as
    ``Diagnosis.dominant_cause`` are emitted as computed fields so consumers of
    the JSON see them without recomputing; on the way back in they are dropped
    and re-derived, so a stored value can never contradict the data it came
    from.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls.model_computed_fields:
            derived = set(cls.model_computed_fields)
            if derived & data.keys():
                return {key: value for key, value in data.items() if key not in derived}
        return data


# --------------------------------------------------------------------------- #
# 1. AgentSpec
# --------------------------------------------------------------------------- #


class OrchestrationStrategy(str, Enum):
    """How the agent loop drives the model. Closed set."""

    SINGLE_SHOT = "single_shot"
    REACT = "react"
    PLAN_THEN_EXECUTE = "plan_then_execute"
    REFLEXION = "reflexion"
    TREE_SEARCH = "tree_search"
    DELEGATING_SUBAGENTS = "delegating_subagents"


class MemoryKind(str, Enum):
    """How context is carried across steps. Closed set."""

    NONE = "none"
    FULL_TRANSCRIPT = "full_transcript"
    SCRATCHPAD = "scratchpad"
    ROLLING_SUMMARY = "rolling_summary"
    VECTOR_RETRIEVAL = "vector_retrieval"
    EPISODIC_STORE = "episodic_store"


_RETRIEVAL_KINDS = frozenset({MemoryKind.VECTOR_RETRIEVAL, MemoryKind.EPISODIC_STORE})


class MemoryConfig(_Frozen):
    """Memory configuration for an :class:`AgentSpec`."""

    kind: MemoryKind = MemoryKind.FULL_TRANSCRIPT
    max_tokens: PositiveInt | None = Field(
        default=None, description="Token budget for retained memory. None means unbounded."
    )
    retrieval_k: PositiveInt | None = Field(
        default=None,
        description="Items fetched per retrieval. Required for, and only for, retrieval kinds.",
    )
    persist_across_runs: bool = False

    @model_validator(mode="after")
    def _check_kind_consistency(self) -> Self:
        if self.kind in _RETRIEVAL_KINDS and self.retrieval_k is None:
            raise ValueError(f"retrieval_k is required for memory kind {self.kind.value}")
        if self.kind not in _RETRIEVAL_KINDS and self.retrieval_k is not None:
            raise ValueError(f"retrieval_k is not meaningful for memory kind {self.kind.value}")
        if self.kind is MemoryKind.NONE and self.max_tokens is not None:
            raise ValueError("max_tokens is not meaningful for memory kind none")
        return self


class StoppingConditions(_Frozen):
    """When the agent loop must halt. At least one bound is required."""

    max_steps: PositiveInt | None = None
    max_tokens: PositiveInt | None = None
    max_wall_clock_seconds: PositiveFloat | None = None
    stop_on_tools: tuple[ToolName, ...] = Field(
        default=(), description="Tool names that terminate the run as soon as they are called."
    )
    require_final_answer: bool = True

    @field_validator("stop_on_tools")
    @classmethod
    def _no_duplicate_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _reject_duplicates(value, "stop_on_tools")

    @model_validator(mode="after")
    def _require_a_bound(self) -> Self:
        bounds = (self.max_steps, self.max_tokens, self.max_wall_clock_seconds)
        if all(bound is None for bound in bounds) and not self.stop_on_tools:
            raise ValueError(
                "StoppingConditions needs at least one of max_steps, max_tokens, "
                "max_wall_clock_seconds, or stop_on_tools"
            )
        return self


def _reject_duplicates(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field_name} contains duplicate tool name {value!r}")
        seen.add(value)
    return values


class AgentSpec(_Frozen):
    """A candidate agent configuration: the unit the engineer loop mutates.

    Two specs describing the same agent compare equal and render to the same
    canonical text, so :func:`diff_agent_specs` between them is empty.
    """

    schema_name: Literal["AgentSpec"] = "AgentSpec"
    schema_version: Literal[1] = SCHEMA_VERSION

    spec_id: NonEmptyStr
    system_prompt: NonEmptyStr
    tools: tuple[ToolName, ...] = Field(
        default=(),
        description="Names of the tools exposed to the agent. Order is preserved and diffed.",
    )
    strategy: OrchestrationStrategy = OrchestrationStrategy.REACT
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    stopping: StoppingConditions = Field(
        default_factory=lambda: StoppingConditions(max_steps=20)
    )
    parent_spec_id: NonEmptyStr | None = Field(
        default=None, description="The spec this one was mutated from, if any."
    )

    @field_validator("system_prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("system_prompt must not be blank")
        return value

    @field_validator("tools")
    @classmethod
    def _no_duplicate_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _reject_duplicates(value, "tools")

    @model_validator(mode="after")
    def _check_internal_references(self) -> Self:
        if self.parent_spec_id == self.spec_id:
            raise ValueError("parent_spec_id must differ from spec_id")
        unknown = sorted(set(self.stopping.stop_on_tools) - set(self.tools))
        if unknown:
            raise ValueError(
                "stopping.stop_on_tools references tools not exposed by the spec: "
                + ", ".join(unknown)
            )
        return self

    def to_text(self) -> str:
        """Render the spec as stable, line-oriented, human-readable text.

        One field per line in a fixed order, so a change to one field produces
        exactly one changed line under ``diff``. The system prompt is emitted
        last, indented, one output line per prompt line, so a multi-line prompt
        stays readable and prompt edits diff at line granularity.
        """
        memory_bits = [f"kind={self.memory.kind.value}"]
        if self.memory.max_tokens is not None:
            memory_bits.append(f"max_tokens={self.memory.max_tokens}")
        if self.memory.retrieval_k is not None:
            memory_bits.append(f"retrieval_k={self.memory.retrieval_k}")
        memory_bits.append(f"persist_across_runs={_render_bool(self.memory.persist_across_runs)}")

        stopping = self.stopping
        lines = [
            f"spec_id: {self.spec_id}",
            f"parent_spec_id: {self.parent_spec_id or '-'}",
            f"strategy: {self.strategy.value}",
            f"memory: {', '.join(memory_bits)}",
            "tools:",
        ]
        lines.extend(f"  - {name}" for name in self.tools) if self.tools else lines.append(
            "  (none)"
        )
        lines += [
            "stopping:",
            f"  max_steps: {_render_optional(stopping.max_steps)}",
            f"  max_tokens: {_render_optional(stopping.max_tokens)}",
            f"  max_wall_clock_seconds: {_render_optional(stopping.max_wall_clock_seconds)}",
            f"  stop_on_tools: {', '.join(stopping.stop_on_tools) or '-'}",
            f"  require_final_answer: {_render_bool(stopping.require_final_answer)}",
            "system_prompt: |",
        ]
        lines.extend(f"  {line}" for line in self.system_prompt.splitlines() or [""])
        return "\n".join(lines) + "\n"


def _render_optional(value: object) -> str:
    return "-" if value is None else str(value)


def _render_bool(value: bool) -> str:
    return "true" if value else "false"


def diff_agent_specs(before: AgentSpec, after: AgentSpec, *, context_lines: int = 3) -> str:
    """Return a human-readable unified diff between two specs.

    Empty string when the two specs render identically. Each side is its
    canonical :meth:`AgentSpec.to_text` form, so a prompt edit shows as changed
    prompt lines and a tool change as an added or removed tool line.
    """
    if not isinstance(before, AgentSpec) or not isinstance(after, AgentSpec):
        raise TypeError("diff_agent_specs requires two AgentSpec instances")
    return "".join(
        difflib.unified_diff(
            before.to_text().splitlines(keepends=True),
            after.to_text().splitlines(keepends=True),
            fromfile=f"a/{before.spec_id}",
            tofile=f"b/{after.spec_id}",
            n=context_lines,
        )
    )


# --------------------------------------------------------------------------- #
# 2. Trajectory
# --------------------------------------------------------------------------- #


class TokenUsage(_Frozen):
    """Token accounting for a run or a single call."""

    prompt_tokens: NonNegativeInt = 0
    completion_tokens: NonNegativeInt = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ToolCall(_Frozen):
    """One tool invocation inside a trajectory, with its arguments and result."""

    ordinal: NonNegativeInt = Field(
        description="0-based position in the trajectory. Must be contiguous and ascending."
    )
    tool_name: ToolName
    args: dict[str, JsonValue] = Field(default_factory=dict)
    result: JsonValue = Field(
        default=None, description="The tool's return value on success; None when error is set."
    )
    error: NonEmptyStr | None = Field(
        default=None, description="Failure message, or None if the call succeeded."
    )
    elapsed_seconds: NonNegativeFloat = 0.0
    tokens: TokenUsage | None = Field(
        default=None, description="Per-call token usage when the harness records it."
    )

    @model_validator(mode="after")
    def _result_xor_error(self) -> Self:
        if self.error is not None and self.result is not None:
            raise ValueError("a ToolCall cannot carry both a result and an error")
        return self

    @property
    def succeeded(self) -> bool:
        return self.error is None


class EvaluatorVerdict(_Frozen):
    """The evaluator's judgement of a trajectory's final answer."""

    evaluator_id: NonEmptyStr
    passed: bool
    score: UnitInterval = 0.0
    rationale: str = ""


class Trajectory(_Frozen):
    """One recorded run of an :class:`AgentSpec` against one task."""

    schema_name: Literal["Trajectory"] = "Trajectory"
    schema_version: Literal[1] = SCHEMA_VERSION

    trajectory_id: NonEmptyStr
    spec_id: NonEmptyStr
    task_id: NonEmptyStr
    tool_calls: tuple[ToolCall, ...] = Field(
        default=(), description="Ordered by ToolCall.ordinal, contiguous from 0."
    )
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_seconds: NonNegativeFloat = Field(
        default=0.0, description="Wall-clock duration of the whole run, including model latency."
    )
    final_answer: str | None = Field(
        default=None, description="None when the run stopped without producing an answer."
    )
    verdict: EvaluatorVerdict | None = Field(
        default=None, description="None when the trajectory has not been evaluated yet."
    )

    @field_validator("tool_calls")
    @classmethod
    def _ordinals_are_contiguous(cls, value: tuple[ToolCall, ...]) -> tuple[ToolCall, ...]:
        for index, call in enumerate(value):
            if call.ordinal != index:
                raise ValueError(
                    "tool_calls must be contiguous and ascending from 0; "
                    f"position {index} has ordinal {call.ordinal}"
                )
        return value

    @property
    def step_count(self) -> int:
        return len(self.tool_calls)

    @property
    def failed(self) -> bool:
        """True when an evaluated trajectory did not pass. Unevaluated runs are not failures."""
        return self.verdict is not None and not self.verdict.passed


# --------------------------------------------------------------------------- #
# 3. Diagnosis
# --------------------------------------------------------------------------- #


class FailureCause(str, Enum):
    """The closed, exhaustive set of causes a failure may be attributed to.

    There is deliberately **no** ``OTHER`` or ``UNKNOWN`` member. A catch-all
    absorbs every hard case, which makes the dominant-cause computation
    meaningless and leaves the mutation stage nothing to aim at. Every member
    below names a failure mode that a mutation can act on, and between them they
    partition the ways an agent run can go wrong -- independently of domain:

    * something went wrong choosing a tool -> :attr:`WRONG_TOOL_SELECTED`
    * ...or calling the tool it chose -> :attr:`TOOL_MISUSE`
    * ...or in the thinking between calls -> :attr:`FAULTY_REASONING`
    * ...or in carrying information forward -> :attr:`CONTEXT_LOSS`
    * ...or in when it stopped: too early -> :attr:`PREMATURE_STOP`,
      or forced out by a budget -> :attr:`STOPPING_CONDITION_HIT`
    * ...or in how the answer was presented -> :attr:`OUTPUT_FORMAT_VIOLATION`

    A failure that seems to fit none of these is a failure that has not been
    analyzed far enough yet. Diagnosis stages must resolve it to one of these
    seven, not park it in a bucket.
    """

    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    """Chose the wrong tool for the step, or reached for none when one was needed.
    Includes needing a capability the spec exposes no tool for."""

    TOOL_MISUSE = "tool_misuse"
    """Called the right tool the wrong way: malformed or wrong arguments, ignoring
    a returned error, or failing to act on what the tool actually said."""

    FAULTY_REASONING = "faulty_reasoning"
    """The thinking between calls was wrong: bad plan or decomposition, an
    unfounded claim, or an invalid inference from correct observations."""

    CONTEXT_LOSS = "context_loss"
    """Information the agent had was not carried to where it was needed --
    dropped from memory, summarized away, or pushed out of the window."""

    PREMATURE_STOP = "premature_stop"
    """Stopped while budget remained: answered early, gave up, or looped without
    progress until it quit."""

    STOPPING_CONDITION_HIT = "stopping_condition_hit"
    """Ran out of a configured budget -- steps, tokens, or wall clock -- before
    the work was finished."""

    OUTPUT_FORMAT_VIOLATION = "output_format_violation"
    """The work was right but the final answer did not meet the required shape,
    structure, or contract."""


_CATCH_ALL_NAMES = frozenset(
    {"OTHER", "UNKNOWN", "MISC", "MISCELLANEOUS", "UNCLASSIFIED", "UNCATEGORIZED", "GENERIC"}
)
"""Names a catch-all would plausibly take. Asserted absent by the test suite."""


class FailureAttribution(_Frozen):
    """One failing trajectory mapped to exactly one cause."""

    trajectory_id: NonEmptyStr
    cause: FailureCause
    evidence: NonEmptyStr = Field(
        description="Why this cause, citing what happened in the trajectory."
    )
    confidence: UnitInterval = Field(
        default=1.0, description="Weights this attribution's vote for the dominant cause."
    )
    step_ordinal: NonNegativeInt | None = Field(
        default=None, description="The tool call where the failure originated, when localizable."
    )


class Diagnosis(_Frozen):
    """Per-failure causes for one spec, plus the aggregate dominant cause.

    ``dominant_cause`` is derived, never asserted: it is the cause with the
    greatest summed confidence across attributions, breaking ties by attribution
    count and then by declaration order in :class:`FailureCause`. It is ``None``
    only when there are no attributions. Because the cause enum has no
    catch-all, the dominant cause always names something a mutation can target.
    """

    schema_name: Literal["Diagnosis"] = "Diagnosis"
    schema_version: Literal[1] = SCHEMA_VERSION

    diagnosis_id: NonEmptyStr
    spec_id: NonEmptyStr
    attributions: tuple[FailureAttribution, ...] = ()
    notes: str = ""

    @field_validator("attributions")
    @classmethod
    def _one_attribution_per_trajectory(
        cls, value: tuple[FailureAttribution, ...]
    ) -> tuple[FailureAttribution, ...]:
        seen: set[str] = set()
        for attribution in value:
            if attribution.trajectory_id in seen:
                raise ValueError(
                    "more than one attribution for trajectory "
                    f"{attribution.trajectory_id!r}; each failure gets exactly one cause"
                )
            seen.add(attribution.trajectory_id)
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dominant_cause(self) -> FailureCause | None:
        if not self.attributions:
            return None
        order = {cause: index for index, cause in enumerate(FailureCause)}
        weight: Counter[FailureCause] = Counter()
        count: Counter[FailureCause] = Counter()
        for attribution in self.attributions:
            weight[attribution.cause] += attribution.confidence
            count[attribution.cause] += 1
        return min(weight, key=lambda cause: (-weight[cause], -count[cause], order[cause]))

    def cause_histogram(self) -> dict[FailureCause, int]:
        """Attribution counts per cause, in :class:`FailureCause` declaration order."""
        counts = Counter(attribution.cause for attribution in self.attributions)
        return {cause: counts[cause] for cause in FailureCause if counts[cause]}


# --------------------------------------------------------------------------- #
# 4. Mutation
# --------------------------------------------------------------------------- #


class MutationKind(str, Enum):
    """The closed set of edits that may be applied to an :class:`AgentSpec`."""

    SYSTEM_PROMPT_REWRITE = "system_prompt_rewrite"
    TOOL_ADDED = "tool_added"
    TOOL_REMOVED = "tool_removed"
    TOOL_SET_REORDERED = "tool_set_reordered"
    STRATEGY_CHANGED = "strategy_changed"
    MEMORY_RECONFIGURED = "memory_reconfigured"
    STOPPING_ADJUSTED = "stopping_adjusted"


class Mutation(_Frozen):
    """One edit to an :class:`AgentSpec`, carrying the :class:`Diagnosis` that motivated it.

    ``motivating_diagnosis`` is the whole Diagnosis, embedded by value, not a
    prose note or a dangling id: given a Mutation you can traverse straight to
    the attributions and evidence behind it, and from there to the failing
    trajectory ids. ``parent_spec_id`` and ``child_spec_id`` close the loop back
    to the spec lineage, so the demo's lineage view is a graph walk, not a
    join against a database that may not have the record.
    """

    schema_name: Literal["Mutation"] = "Mutation"
    schema_version: Literal[1] = SCHEMA_VERSION

    mutation_id: NonEmptyStr
    kind: MutationKind
    target_path: NonEmptyStr = Field(
        description="Dotted field path into AgentSpec, e.g. 'system_prompt' or 'memory.kind'."
    )
    before: str = Field(description="Prior value rendered as text. Empty for a pure addition.")
    after: str = Field(description="New value rendered as text. Empty for a pure removal.")
    rationale: NonEmptyStr = Field(
        description="Why this edit should fix the motivating cause."
    )
    motivating_diagnosis: Diagnosis = Field(
        description="The Diagnosis this edit answers, embedded so the link is traversable."
    )
    motivating_cause: FailureCause = Field(
        description="Which cause from the motivating Diagnosis this edit targets."
    )
    parent_spec_id: NonEmptyStr
    child_spec_id: NonEmptyStr | None = Field(
        default=None, description="The spec produced by applying this mutation, once it exists."
    )

    @property
    def motivating_diagnosis_id(self) -> str:
        """Convenience accessor for the embedded diagnosis's id."""
        return self.motivating_diagnosis.diagnosis_id

    @model_validator(mode="after")
    def _check_links(self) -> Self:
        if self.before == self.after:
            raise ValueError("before and after must differ; a mutation must change something")
        if self.child_spec_id is not None and self.child_spec_id == self.parent_spec_id:
            raise ValueError("child_spec_id must differ from parent_spec_id")
        if self.motivating_diagnosis.spec_id != self.parent_spec_id:
            raise ValueError(
                f"motivating_diagnosis diagnoses spec {self.motivating_diagnosis.spec_id!r} "
                f"but this mutation edits spec {self.parent_spec_id!r}"
            )
        attributed = {attribution.cause for attribution in self.motivating_diagnosis.attributions}
        if attributed and self.motivating_cause not in attributed:
            raise ValueError(
                f"motivating_cause {self.motivating_cause.value!r} is not among the causes "
                "attributed by the motivating diagnosis: "
                + ", ".join(sorted(cause.value for cause in attributed))
            )
        return self

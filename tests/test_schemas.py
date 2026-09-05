"""Round-trip, validation, closed-enum, and diff tests for the four frozen schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_engineer.schemas import (
    AgentSpec,
    Diagnosis,
    EvaluatorVerdict,
    FailureAttribution,
    FailureCause,
    MemoryConfig,
    MemoryKind,
    Mutation,
    MutationKind,
    OrchestrationStrategy,
    StoppingConditions,
    TokenUsage,
    ToolCall,
    Trajectory,
    diff_agent_specs,
)
from agent_engineer.schemas import _CATCH_ALL_NAMES

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def make_spec(**overrides) -> AgentSpec:
    base = dict(
        spec_id="spec-001",
        system_prompt="You are a capable agent.\nAlways cite the evidence you used.",
        tools=("search", "fetch", "submit_answer"),
        strategy=OrchestrationStrategy.REACT,
        memory=MemoryConfig(kind=MemoryKind.ROLLING_SUMMARY, max_tokens=4096),
        stopping=StoppingConditions(
            max_steps=12,
            max_tokens=100_000,
            max_wall_clock_seconds=90.0,
            stop_on_tools=("submit_answer",),
        ),
    )
    base.update(overrides)
    return AgentSpec(**base)


def make_trajectory(**overrides) -> Trajectory:
    base = dict(
        trajectory_id="traj-001",
        spec_id="spec-001",
        task_id="task-042",
        tool_calls=(
            ToolCall(
                ordinal=0,
                tool_name="search",
                args={"query": "example", "limit": 3},
                result=[{"id": "a", "url": "https://example.test/a"}],
                elapsed_seconds=0.8,
                tokens=TokenUsage(prompt_tokens=120, completion_tokens=18),
            ),
            ToolCall(
                ordinal=1,
                tool_name="fetch",
                args={"url": "https://example.test/a"},
                error="404 Not Found",
                elapsed_seconds=0.2,
            ),
            ToolCall(
                ordinal=2,
                tool_name="submit_answer",
                args={"answer": "A"},
                result={"accepted": True, "nested": {"depth": [1, 2, 3]}},
                elapsed_seconds=0.05,
            ),
        ),
        tokens=TokenUsage(prompt_tokens=2100, completion_tokens=340),
        elapsed_seconds=6.75,
        final_answer="A",
        verdict=EvaluatorVerdict(
            evaluator_id="exact-match-v2",
            passed=True,
            score=1.0,
            rationale="Matches the reference answer.",
        ),
    )
    base.update(overrides)
    return Trajectory(**base)


def make_diagnosis(**overrides) -> Diagnosis:
    base = dict(
        diagnosis_id="diag-001",
        spec_id="spec-001",
        attributions=(
            FailureAttribution(
                trajectory_id="traj-001",
                cause=FailureCause.TOOL_MISUSE,
                evidence="fetch returned 404 and the agent answered from the error page anyway.",
                confidence=0.9,
                step_ordinal=1,
            ),
            FailureAttribution(
                trajectory_id="traj-002",
                cause=FailureCause.TOOL_MISUSE,
                evidence="Same unhandled 404 against a different argument.",
                confidence=0.6,
            ),
            FailureAttribution(
                trajectory_id="traj-003",
                cause=FailureCause.STOPPING_CONDITION_HIT,
                evidence="Hit max_steps=12 mid-plan.",
                confidence=1.0,
            ),
        ),
        notes="Three failures over the held-out suite.",
    )
    base.update(overrides)
    return Diagnosis(**base)


def make_mutation(**overrides) -> Mutation:
    base = dict(
        mutation_id="mut-001",
        kind=MutationKind.SYSTEM_PROMPT_REWRITE,
        target_path="system_prompt",
        before="You are a capable agent.",
        after="You are a capable agent. If a tool returns an error, do not use its output.",
        rationale="Tool misuse dominates; state an explicit error-handling rule.",
        motivating_diagnosis=make_diagnosis(),
        motivating_cause=FailureCause.TOOL_MISUSE,
        parent_spec_id="spec-001",
        child_spec_id="spec-002",
    )
    base.update(overrides)
    return Mutation(**base)


# --------------------------------------------------------------------------- #
# round-trip serialization (JSON is the wire format)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "obj",
    [
        make_spec(),
        make_spec(tools=(), stopping=StoppingConditions(max_steps=1), parent_spec_id="spec-000"),
        make_spec(
            strategy=OrchestrationStrategy.TREE_SEARCH,
            memory=MemoryConfig(
                kind=MemoryKind.VECTOR_RETRIEVAL, retrieval_k=8, persist_across_runs=True
            ),
        ),
        make_spec(memory=MemoryConfig(kind=MemoryKind.NONE)),
    ],
    ids=["full", "minimal", "retrieval-memory", "no-memory"],
)
def test_agent_spec_round_trips(obj: AgentSpec) -> None:
    assert AgentSpec.model_validate(obj.model_dump()) == obj
    assert AgentSpec.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize(
    "obj",
    [
        make_trajectory(),
        make_trajectory(tool_calls=(), final_answer=None, verdict=None),
        make_trajectory(
            final_answer="",
            verdict=EvaluatorVerdict(evaluator_id="llm-judge", passed=False, score=0.25),
        ),
    ],
    ids=["full", "empty", "failed"],
)
def test_trajectory_round_trips(obj: Trajectory) -> None:
    assert Trajectory.model_validate(obj.model_dump()) == obj
    assert Trajectory.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize(
    "obj",
    [make_diagnosis(), make_diagnosis(attributions=(), notes="")],
    ids=["full", "empty"],
)
def test_diagnosis_round_trips(obj: Diagnosis) -> None:
    assert Diagnosis.model_validate(obj.model_dump()) == obj
    assert Diagnosis.model_validate_json(obj.model_dump_json()) == obj


@pytest.mark.parametrize(
    "obj",
    [
        make_mutation(),
        make_mutation(
            mutation_id="mut-002",
            kind=MutationKind.STOPPING_ADJUSTED,
            target_path="stopping.max_steps",
            before="12",
            after="30",
            motivating_cause=FailureCause.STOPPING_CONDITION_HIT,
            child_spec_id=None,
        ),
    ],
    ids=["prompt-rewrite", "stopping-adjusted"],
)
def test_mutation_round_trips(obj: Mutation) -> None:
    assert Mutation.model_validate(obj.model_dump()) == obj
    assert Mutation.model_validate_json(obj.model_dump_json()) == obj


def test_round_trip_is_idempotent_and_byte_stable() -> None:
    for obj, cls in [
        (make_spec(), AgentSpec),
        (make_trajectory(), Trajectory),
        (make_diagnosis(), Diagnosis),
        (make_mutation(), Mutation),
    ]:
        once = obj.model_dump_json()
        twice = cls.model_validate_json(once).model_dump_json()
        assert once == twice, f"{cls.__name__} serialization is not stable"


def test_json_is_deterministic_regardless_of_input_key_order() -> None:
    spec = make_spec()
    shuffled = dict(reversed(list(json.loads(spec.model_dump_json()).items())))
    assert AgentSpec.model_validate(shuffled).model_dump_json() == spec.model_dump_json()


def test_wire_payload_carries_schema_identity() -> None:
    payload = json.loads(make_spec().model_dump_json())
    assert payload["schema_name"] == "AgentSpec"
    assert payload["schema_version"] == 1


def test_tool_call_preserves_nested_args_and_results() -> None:
    call = ToolCall(
        ordinal=0,
        tool_name="query",
        args={"filters": {"b": 2, "a": [1, {"deep": None}]}, "flag": True},
        result={"rows": [[1, 2], []], "count": 0},
    )
    decoded = ToolCall.model_validate_json(call.model_dump_json())
    assert decoded == call
    assert decoded.args["filters"]["a"][1]["deep"] is None
    assert decoded.result["rows"] == [[1, 2], []]


# --------------------------------------------------------------------------- #
# AgentSpec human-readable text + diff
# --------------------------------------------------------------------------- #


def test_diff_is_empty_for_equivalent_specs() -> None:
    assert diff_agent_specs(make_spec(), make_spec()) == ""


def test_diff_shows_prompt_edit_at_line_granularity() -> None:
    before = make_spec()
    after = before.model_copy(
        update={"system_prompt": "You are a capable agent.\nAlways cite two independent sources."}
    )
    diff = diff_agent_specs(before, after)
    assert "-  Always cite the evidence you used." in diff
    assert "+  Always cite two independent sources." in diff
    assert "  You are a capable agent." in diff
    # Untouched fields do not appear as changes.
    assert "-strategy:" not in diff


def test_diff_shows_tool_and_strategy_changes() -> None:
    before = make_spec()
    after = before.model_copy(
        update={
            "tools": ("search", "fetch", "calculator", "submit_answer"),
            "strategy": OrchestrationStrategy.PLAN_THEN_EXECUTE,
        }
    )
    diff = diff_agent_specs(before, after)
    assert "+  - calculator" in diff
    assert "-strategy: react" in diff
    assert "+strategy: plan_then_execute" in diff


def test_diff_shows_memory_and_stopping_changes() -> None:
    before = make_spec()
    after = before.model_copy(
        update={
            "memory": MemoryConfig(kind=MemoryKind.VECTOR_RETRIEVAL, retrieval_k=5),
            "stopping": StoppingConditions(max_steps=30, stop_on_tools=("submit_answer",)),
        }
    )
    diff = diff_agent_specs(before, after)
    assert "+memory: kind=vector_retrieval, retrieval_k=5, persist_across_runs=false" in diff
    assert "-  max_steps: 12" in diff
    assert "+  max_steps: 30" in diff


def test_to_text_is_stable_one_field_per_line_and_prompt_readable() -> None:
    text = make_spec().to_text()
    assert text == make_spec().to_text()
    assert text.endswith("\n")
    lines = text.splitlines()
    assert lines[0] == "spec_id: spec-001"
    assert lines.index("system_prompt: |") == len(lines) - 3
    # The multi-line prompt stays readable: one indented line per prompt line.
    assert lines[-2:] == [
        "  You are a capable agent.",
        "  Always cite the evidence you used.",
    ]


def test_diff_rejects_non_specs() -> None:
    with pytest.raises(TypeError):
        diff_agent_specs(make_spec(), "not a spec")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the cause enum is CLOSED -- no catch-all
# --------------------------------------------------------------------------- #


def test_failure_cause_has_no_catch_all_member() -> None:
    """A catch-all would absorb every hard case and void the dominant-cause signal."""
    for member in FailureCause:
        assert member.name not in _CATCH_ALL_NAMES, f"{member.name} is a catch-all bucket"
        assert member.value not in {name.lower() for name in _CATCH_ALL_NAMES}
        for banned in ("other", "unknown", "misc", "unclassified", "uncategorized"):
            assert banned not in member.value, f"{member.value} reads as a catch-all"


def test_failure_cause_is_small_and_exhaustive_by_design() -> None:
    assert 5 <= len(FailureCause) <= 8
    assert {member.value for member in FailureCause} == {
        "wrong_tool_selected",
        "tool_misuse",
        "faulty_reasoning",
        "context_loss",
        "premature_stop",
        "stopping_condition_hit",
        "output_format_violation",
    }


def test_unrecognized_cause_is_rejected_rather_than_bucketed() -> None:
    with pytest.raises(ValidationError):
        FailureAttribution.model_validate(
            {"trajectory_id": "t", "cause": "something_else", "evidence": "e"}
        )


def test_every_cause_is_usable_as_a_mutation_target() -> None:
    """No member is decorative: each can motivate a mutation."""
    for cause in FailureCause:
        diagnosis = make_diagnosis(
            attributions=(FailureAttribution(trajectory_id="t1", cause=cause, evidence="e"),)
        )
        mutation = make_mutation(motivating_diagnosis=diagnosis, motivating_cause=cause)
        assert mutation.motivating_cause is cause


# --------------------------------------------------------------------------- #
# Diagnosis aggregation
# --------------------------------------------------------------------------- #


def test_dominant_cause_is_highest_summed_confidence() -> None:
    # TOOL_MISUSE sums to 1.5 vs STOPPING_CONDITION_HIT at 1.0.
    assert make_diagnosis().dominant_cause is FailureCause.TOOL_MISUSE


def test_dominant_cause_breaks_ties_by_count_then_declaration_order() -> None:
    by_count = Diagnosis(
        diagnosis_id="d",
        spec_id="s",
        attributions=(
            FailureAttribution(trajectory_id="t1", cause=FailureCause.PREMATURE_STOP, evidence="e", confidence=0.5),
            FailureAttribution(trajectory_id="t2", cause=FailureCause.PREMATURE_STOP, evidence="e", confidence=0.5),
            FailureAttribution(trajectory_id="t3", cause=FailureCause.WRONG_TOOL_SELECTED, evidence="e", confidence=1.0),
        ),
    )
    assert by_count.dominant_cause is FailureCause.PREMATURE_STOP

    by_order = Diagnosis(
        diagnosis_id="d",
        spec_id="s",
        attributions=(
            FailureAttribution(trajectory_id="t1", cause=FailureCause.PREMATURE_STOP, evidence="e"),
            FailureAttribution(trajectory_id="t2", cause=FailureCause.WRONG_TOOL_SELECTED, evidence="e"),
        ),
    )
    assert by_order.dominant_cause is FailureCause.WRONG_TOOL_SELECTED


def test_dominant_cause_is_none_without_attributions() -> None:
    empty = Diagnosis(diagnosis_id="d", spec_id="s")
    assert empty.dominant_cause is None
    assert json.loads(empty.model_dump_json())["dominant_cause"] is None


def test_dominant_cause_is_serialized_for_downstream_stages() -> None:
    assert json.loads(make_diagnosis().model_dump_json())["dominant_cause"] == "tool_misuse"


def test_stored_dominant_cause_is_re_derived_not_trusted() -> None:
    payload = json.loads(make_diagnosis().model_dump_json())
    payload["dominant_cause"] = "context_loss"
    assert Diagnosis.model_validate(payload).dominant_cause is FailureCause.TOOL_MISUSE


def test_cause_histogram_follows_enum_order() -> None:
    histogram = make_diagnosis().cause_histogram()
    assert list(histogram) == [FailureCause.TOOL_MISUSE, FailureCause.STOPPING_CONDITION_HIT]
    assert histogram[FailureCause.TOOL_MISUSE] == 2


def test_one_attribution_per_trajectory() -> None:
    with pytest.raises(ValidationError, match="more than one attribution"):
        Diagnosis(
            diagnosis_id="d",
            spec_id="s",
            attributions=(
                FailureAttribution(trajectory_id="t1", cause=FailureCause.CONTEXT_LOSS, evidence="e"),
                FailureAttribution(trajectory_id="t1", cause=FailureCause.FAULTY_REASONING, evidence="e"),
            ),
        )


# --------------------------------------------------------------------------- #
# Mutation -> Diagnosis lineage
# --------------------------------------------------------------------------- #


def test_mutation_carries_a_traversable_diagnosis_reference() -> None:
    mutation = make_mutation()
    assert isinstance(mutation.motivating_diagnosis, Diagnosis)
    assert mutation.motivating_diagnosis_id == "diag-001"
    # Traversable all the way down to the failing trajectory ids, with no lookup.
    assert [a.trajectory_id for a in mutation.motivating_diagnosis.attributions] == [
        "traj-001",
        "traj-002",
        "traj-003",
    ]
    assert mutation.motivating_diagnosis.dominant_cause is mutation.motivating_cause


def test_diagnosis_link_survives_a_json_round_trip() -> None:
    decoded = Mutation.model_validate_json(make_mutation().model_dump_json())
    assert decoded.motivating_diagnosis == make_diagnosis()
    assert decoded.motivating_diagnosis.attributions[0].evidence.startswith("fetch returned 404")


def test_mutation_requires_a_diagnosis() -> None:
    payload = json.loads(make_mutation().model_dump_json())
    del payload["motivating_diagnosis"]
    with pytest.raises(ValidationError, match="motivating_diagnosis"):
        Mutation.model_validate(payload)


def test_mutation_cause_must_come_from_its_diagnosis() -> None:
    with pytest.raises(ValidationError, match="not among the causes attributed"):
        make_mutation(motivating_cause=FailureCause.CONTEXT_LOSS)


def test_mutation_must_diagnose_the_spec_it_edits() -> None:
    with pytest.raises(ValidationError, match="but this mutation edits spec"):
        make_mutation(motivating_diagnosis=make_diagnosis(spec_id="spec-999"))


def test_lineage_walks_from_mutation_to_child_spec() -> None:
    spec = make_spec()
    mutation = make_mutation(parent_spec_id=spec.spec_id, child_spec_id="spec-002")
    child = spec.model_copy(
        update={"spec_id": mutation.child_spec_id, "parent_spec_id": spec.spec_id}
    )
    assert mutation.motivating_diagnosis.spec_id == mutation.parent_spec_id == spec.spec_id
    assert child.parent_spec_id == mutation.parent_spec_id
    assert diff_agent_specs(spec, child).startswith("---")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def test_models_are_frozen_and_hashable() -> None:
    spec = make_spec()
    with pytest.raises(ValidationError):
        spec.spec_id = "other"  # type: ignore[misc]
    assert hash(spec) == hash(make_spec())


def test_models_reject_unknown_fields() -> None:
    payload = json.loads(make_spec().model_dump_json())
    payload["temperature"] = 0.7
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSpec.model_validate(payload)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spec_id": ""}, "at least 1 character"),
        ({"system_prompt": "  "}, "must not be blank"),
        ({"tools": ("a", "a", "submit_answer")}, "duplicate tool name"),
        ({"tools": ("search",)}, "not exposed by the spec"),
        ({"parent_spec_id": "spec-001"}, "must differ from spec_id"),
    ],
)
def test_agent_spec_rejects_invalid_input(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        make_spec(**kwargs)


def test_stopping_conditions_require_a_bound() -> None:
    with pytest.raises(ValidationError, match="at least one of"):
        StoppingConditions()


def test_memory_config_retrieval_k_is_kind_specific() -> None:
    with pytest.raises(ValidationError, match="required for memory kind"):
        MemoryConfig(kind=MemoryKind.VECTOR_RETRIEVAL)
    with pytest.raises(ValidationError, match="not meaningful for memory kind"):
        MemoryConfig(kind=MemoryKind.SCRATCHPAD, retrieval_k=4)


def test_trajectory_requires_contiguous_ordinals() -> None:
    with pytest.raises(ValidationError, match="contiguous and ascending"):
        make_trajectory(
            tool_calls=(
                ToolCall(ordinal=0, tool_name="a"),
                ToolCall(ordinal=2, tool_name="b"),
            )
        )


def test_tool_call_rejects_result_and_error_together() -> None:
    with pytest.raises(ValidationError, match="both a result and an error"):
        ToolCall(ordinal=0, tool_name="a", result="ok", error="boom")


def test_trajectory_derived_properties() -> None:
    traj = make_trajectory()
    assert traj.step_count == 3
    assert traj.failed is False
    assert traj.tokens.total_tokens == 2440
    assert traj.tool_calls[1].succeeded is False
    assert make_trajectory(verdict=None).failed is False
    assert make_trajectory(verdict=EvaluatorVerdict(evaluator_id="e", passed=False)).failed is True


def test_verdict_score_must_be_normalized() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        EvaluatorVerdict(evaluator_id="e", passed=True, score=1.5)


def test_mutation_requires_an_actual_change() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        make_mutation(before="same", after="same")


def test_missing_required_field_is_reported_by_name() -> None:
    payload = json.loads(make_spec().model_dump_json())
    del payload["system_prompt"]
    with pytest.raises(ValidationError, match="system_prompt"):
        AgentSpec.model_validate(payload)

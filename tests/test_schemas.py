"""Round-trip, validation, and diff tests for the four frozen schemas."""

from __future__ import annotations

import json

import pytest

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
    SchemaError,
    StoppingConditions,
    TokenUsage,
    ToolCall,
    Trajectory,
    diff_agent_specs,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def make_spec(**overrides) -> AgentSpec:
    base = dict(
        spec_id="spec-001",
        system_prompt="You are a research agent.\nAlways cite your sources.",
        tools=("web_search", "read_page", "submit_answer"),
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
                tool_name="web_search",
                args={"query": "capital of Peru", "limit": 3},
                result=[{"title": "Lima", "url": "https://example.test/lima"}],
                elapsed_seconds=0.8,
                tokens=TokenUsage(prompt_tokens=120, completion_tokens=18),
            ),
            ToolCall(
                ordinal=1,
                tool_name="read_page",
                args={"url": "https://example.test/lima"},
                error="404 Not Found",
                elapsed_seconds=0.2,
            ),
            ToolCall(
                ordinal=2,
                tool_name="submit_answer",
                args={"answer": "Lima"},
                result={"accepted": True, "nested": {"depth": [1, 2, 3]}},
                elapsed_seconds=0.05,
            ),
        ),
        tokens=TokenUsage(prompt_tokens=2100, completion_tokens=340),
        elapsed_seconds=6.75,
        final_answer="Lima",
        verdict=EvaluatorVerdict(
            evaluator_id="exact-match-v2",
            passed=True,
            score=1.0,
            rationale="Matches the gold answer.",
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
                cause=FailureCause.TOOL_ERROR_UNHANDLED,
                evidence="read_page returned 404 and the agent submitted anyway.",
                confidence=0.9,
                step_ordinal=1,
            ),
            FailureAttribution(
                trajectory_id="traj-002",
                cause=FailureCause.TOOL_ERROR_UNHANDLED,
                evidence="Same unhandled 404 on a different URL.",
                confidence=0.6,
            ),
            FailureAttribution(
                trajectory_id="traj-003",
                cause=FailureCause.BUDGET_EXHAUSTED,
                evidence="Hit max_steps=12 mid-plan.",
                confidence=1.0,
            ),
        ),
        notes="Three failures over the retrieval suite.",
    )
    base.update(overrides)
    return Diagnosis(**base)


def make_mutation(**overrides) -> Mutation:
    base = dict(
        mutation_id="mut-001",
        kind=MutationKind.SYSTEM_PROMPT_REWRITE,
        target_path="system_prompt",
        before="You are a research agent.",
        after="You are a research agent. If a tool errors, retry once with a different source.",
        rationale="Unhandled tool errors dominate; give the agent an explicit recovery rule.",
        motivating_diagnosis_id="diag-001",
        motivating_cause=FailureCause.TOOL_ERROR_UNHANDLED,
        parent_spec_id="spec-001",
        child_spec_id="spec-002",
    )
    base.update(overrides)
    return Mutation(**base)


# --------------------------------------------------------------------------- #
# round-trip serialization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "obj",
    [
        make_spec(),
        make_spec(tools=(), stopping=StoppingConditions(max_steps=1), parent_spec_id="spec-000"),
        make_spec(
            strategy=OrchestrationStrategy.TREE_SEARCH,
            memory=MemoryConfig(kind=MemoryKind.VECTOR_RETRIEVAL, retrieval_k=8, persist_across_runs=True),
        ),
        make_spec(memory=MemoryConfig(kind=MemoryKind.NONE)),
    ],
    ids=["full", "minimal", "retrieval-memory", "no-memory"],
)
def test_agent_spec_round_trips(obj: AgentSpec) -> None:
    assert AgentSpec.from_dict(obj.to_dict()) == obj
    assert AgentSpec.from_json(obj.to_json()) == obj


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
    assert Trajectory.from_dict(obj.to_dict()) == obj
    assert Trajectory.from_json(obj.to_json()) == obj


@pytest.mark.parametrize(
    "obj",
    [make_diagnosis(), make_diagnosis(attributions=(), notes="")],
    ids=["full", "empty"],
)
def test_diagnosis_round_trips(obj: Diagnosis) -> None:
    assert Diagnosis.from_dict(obj.to_dict()) == obj
    assert Diagnosis.from_json(obj.to_json()) == obj


@pytest.mark.parametrize(
    "obj",
    [
        make_mutation(),
        make_mutation(
            mutation_id="mut-002",
            kind=MutationKind.TOOL_ADDED,
            target_path="tools",
            before="",
            after="calculator",
            motivating_cause=FailureCause.MISSING_TOOL,
            child_spec_id=None,
        ),
    ],
    ids=["prompt-rewrite", "tool-added"],
)
def test_mutation_round_trips(obj: Mutation) -> None:
    assert Mutation.from_dict(obj.to_dict()) == obj
    assert Mutation.from_json(obj.to_json()) == obj


def test_round_trip_is_idempotent_and_byte_stable() -> None:
    for obj, cls in [
        (make_spec(), AgentSpec),
        (make_trajectory(), Trajectory),
        (make_diagnosis(), Diagnosis),
        (make_mutation(), Mutation),
    ]:
        once = obj.to_json()
        twice = cls.from_json(once).to_json()
        assert once == twice, f"{cls.__name__} serialization is not stable"


def test_json_is_deterministic_regardless_of_input_key_order() -> None:
    spec = make_spec()
    shuffled = dict(reversed(list(spec.to_dict().items())))
    assert AgentSpec.from_dict(shuffled).to_json() == spec.to_json()


def test_tool_call_preserves_nested_args_and_results() -> None:
    call = ToolCall(
        ordinal=0,
        tool_name="query",
        args={"filters": {"b": 2, "a": [1, {"deep": None}]}, "flag": True},
        result={"rows": [[1, 2], []], "count": 0},
    )
    decoded = ToolCall.from_dict(json.loads(json.dumps(call.to_dict())))
    assert decoded == call
    assert decoded.args["filters"]["a"][1]["deep"] is None
    # Stored nested values are immutable; to_dict() thaws them back to JSON types.
    assert decoded.result["rows"] == ((1, 2), ())
    assert decoded.to_dict()["result"] == {"rows": [[1, 2], []], "count": 0}


# --------------------------------------------------------------------------- #
# AgentSpec text + diff
# --------------------------------------------------------------------------- #


def test_diff_is_empty_for_equivalent_specs() -> None:
    assert diff_agent_specs(make_spec(), make_spec()) == ""


def test_diff_shows_prompt_edit_at_line_granularity() -> None:
    before = make_spec()
    after = before.with_changes(
        system_prompt="You are a research agent.\nAlways cite two independent sources."
    )
    diff = diff_agent_specs(before, after)
    assert "-  Always cite your sources." in diff
    assert "+  Always cite two independent sources." in diff
    assert "You are a research agent." in diff
    # Untouched fields do not show up as changes.
    assert "-strategy:" not in diff


def test_diff_shows_tool_and_strategy_changes() -> None:
    before = make_spec()
    after = before.with_changes(
        tools=("web_search", "read_page", "calculator", "submit_answer"),
        strategy=OrchestrationStrategy.PLAN_THEN_EXECUTE,
    )
    diff = diff_agent_specs(before, after)
    assert "+  - calculator" in diff
    assert "-strategy: react" in diff
    assert "+strategy: plan_then_execute" in diff


def test_diff_shows_memory_and_stopping_changes() -> None:
    before = make_spec()
    after = before.with_changes(
        memory=MemoryConfig(kind=MemoryKind.VECTOR_RETRIEVAL, retrieval_k=5),
        stopping=StoppingConditions(max_steps=30, stop_on_tools=("submit_answer",)),
    )
    diff = diff_agent_specs(before, after)
    assert "+memory: kind=vector_retrieval, retrieval_k=5, persist_across_runs=false" in diff
    assert "-  max_steps: 12" in diff
    assert "+  max_steps: 30" in diff


def test_to_text_is_stable_and_ends_with_newline() -> None:
    text = make_spec().to_text()
    assert text.endswith("\n")
    assert text == make_spec().to_text()
    assert text.splitlines()[0] == "spec_id: spec-001"


def test_diff_rejects_non_specs() -> None:
    with pytest.raises(SchemaError):
        diff_agent_specs(make_spec(), "not a spec")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Diagnosis aggregation over the closed cause enum
# --------------------------------------------------------------------------- #


def test_dominant_cause_is_highest_summed_confidence() -> None:
    # TOOL_ERROR_UNHANDLED sums to 1.5 vs BUDGET_EXHAUSTED at 1.0.
    assert make_diagnosis().dominant_cause is FailureCause.TOOL_ERROR_UNHANDLED


def test_dominant_cause_breaks_ties_by_count_then_declaration_order() -> None:
    by_count = Diagnosis(
        diagnosis_id="d",
        spec_id="s",
        attributions=(
            FailureAttribution("t1", FailureCause.PREMATURE_STOP, "e", confidence=0.5),
            FailureAttribution("t2", FailureCause.PREMATURE_STOP, "e", confidence=0.5),
            FailureAttribution("t3", FailureCause.MISSING_TOOL, "e", confidence=1.0),
        ),
    )
    assert by_count.dominant_cause is FailureCause.PREMATURE_STOP

    by_order = Diagnosis(
        diagnosis_id="d",
        spec_id="s",
        attributions=(
            FailureAttribution("t1", FailureCause.PREMATURE_STOP, "e"),
            FailureAttribution("t2", FailureCause.MISSING_TOOL, "e"),
        ),
    )
    assert by_order.dominant_cause is FailureCause.MISSING_TOOL


def test_dominant_cause_is_none_without_attributions() -> None:
    empty = Diagnosis(diagnosis_id="d", spec_id="s")
    assert empty.dominant_cause is None
    assert empty.to_dict()["dominant_cause"] is None


def test_dominant_cause_is_serialized_and_verified_on_decode() -> None:
    payload = make_diagnosis().to_dict()
    assert payload["dominant_cause"] == "tool_error_unhandled"
    payload["dominant_cause"] = "hallucinated_fact"
    with pytest.raises(SchemaError, match="disagrees with the attributions"):
        Diagnosis.from_dict(payload)


def test_cause_histogram_follows_enum_order() -> None:
    histogram = make_diagnosis().cause_histogram()
    assert list(histogram) == [FailureCause.TOOL_ERROR_UNHANDLED, FailureCause.BUDGET_EXHAUSTED]
    assert histogram[FailureCause.TOOL_ERROR_UNHANDLED] == 2


def test_cause_enum_is_closed() -> None:
    with pytest.raises(SchemaError, match="must be one of"):
        FailureAttribution.from_dict(
            {"trajectory_id": "t", "cause": "agent_was_sleepy", "evidence": "e"}
        )


def test_one_attribution_per_trajectory() -> None:
    with pytest.raises(SchemaError, match="more than one attribution"):
        Diagnosis(
            diagnosis_id="d",
            spec_id="s",
            attributions=(
                FailureAttribution("t1", FailureCause.MEMORY_LOSS, "e"),
                FailureAttribution("t1", FailureCause.REPEATED_LOOP, "e"),
            ),
        )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def test_schemas_are_frozen_and_hashable() -> None:
    spec = make_spec()
    with pytest.raises(Exception):
        spec.spec_id = "other"  # type: ignore[misc]
    assert hash(spec) == hash(make_spec())
    assert hash(make_mutation()) == hash(make_mutation())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spec_id": ""}, "must be non-empty"),
        ({"system_prompt": "  "}, "must be non-empty"),
        ({"tools": ("a", "a", "submit_answer")}, "duplicate tool name"),
        ({"tools": ("web_search",)}, "not exposed by the spec"),
        ({"parent_spec_id": "spec-001"}, "must differ from spec_id"),
    ],
)
def test_agent_spec_rejects_invalid_input(kwargs, message) -> None:
    with pytest.raises(SchemaError, match=message):
        make_spec(**kwargs)


def test_stopping_conditions_require_a_bound() -> None:
    with pytest.raises(SchemaError, match="at least one of"):
        StoppingConditions()


def test_memory_config_retrieval_k_is_kind_specific() -> None:
    with pytest.raises(SchemaError, match="required for kind"):
        MemoryConfig(kind=MemoryKind.VECTOR_RETRIEVAL)
    with pytest.raises(SchemaError, match="not meaningful for kind"):
        MemoryConfig(kind=MemoryKind.SCRATCHPAD, retrieval_k=4)


def test_trajectory_requires_contiguous_ordinals() -> None:
    with pytest.raises(SchemaError, match="contiguous and ascending"):
        make_trajectory(
            tool_calls=(
                ToolCall(ordinal=0, tool_name="a"),
                ToolCall(ordinal=2, tool_name="b"),
            )
        )


def test_tool_call_rejects_result_and_error_together() -> None:
    with pytest.raises(SchemaError, match="both a result and an error"):
        ToolCall(ordinal=0, tool_name="a", result="ok", error="boom")


def test_tool_call_rejects_non_json_args() -> None:
    with pytest.raises(SchemaError, match="JSON-representable"):
        ToolCall(ordinal=0, tool_name="a", args={"when": object()})


def test_trajectory_derived_properties() -> None:
    traj = make_trajectory()
    assert traj.step_count == 3
    assert traj.failed is False
    assert traj.tokens.total_tokens == 2440
    assert traj.tool_calls[1].succeeded is False
    assert make_trajectory(verdict=None).failed is False
    assert make_trajectory(
        verdict=EvaluatorVerdict(evaluator_id="e", passed=False)
    ).failed is True


def test_verdict_score_must_be_normalized() -> None:
    with pytest.raises(SchemaError, match=r"within \[0.0, 1.0\]"):
        EvaluatorVerdict(evaluator_id="e", passed=True, score=1.5)


def test_mutation_requires_an_actual_change() -> None:
    with pytest.raises(SchemaError, match="must differ"):
        make_mutation(before="same", after="same")


def test_mutation_requires_a_motivating_diagnosis() -> None:
    with pytest.raises(SchemaError, match="motivating_diagnosis_id"):
        make_mutation(motivating_diagnosis_id="")


def test_missing_required_key_is_reported_by_name() -> None:
    payload = make_spec().to_dict()
    del payload["system_prompt"]
    with pytest.raises(SchemaError, match="missing required key 'system_prompt'"):
        AgentSpec.from_dict(payload)


def test_mutation_links_diagnosis_to_spec_lineage() -> None:
    diagnosis = make_diagnosis()
    spec = make_spec()
    mutation = make_mutation(
        motivating_diagnosis_id=diagnosis.diagnosis_id,
        motivating_cause=diagnosis.dominant_cause,
        parent_spec_id=spec.spec_id,
        child_spec_id="spec-002",
    )
    child = spec.with_changes(spec_id=mutation.child_spec_id, parent_spec_id=spec.spec_id)
    assert mutation.motivating_cause is FailureCause.TOOL_ERROR_UNHANDLED
    assert child.parent_spec_id == mutation.parent_spec_id
    assert diff_agent_specs(spec, child).startswith("---")

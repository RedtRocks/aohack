"""Tests for agent_engineer.memory: episodic store, self-reflection, and retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_engineer.memory import (
    EpisodicEntry,
    EpisodicMemoryStore,
    HeuristicReflector,
    ModelReflector,
    distill_reflections,
)
from agent_engineer.schemas import (
    AgentSpec,
    Diagnosis,
    EvaluatorVerdict,
    FailureAttribution,
    FailureCause,
    MemoryConfig,
    MemoryKind,
    StoppingConditions,
    ToolCall,
    Trajectory,
)
from agent_engineer.stages.evaluate import EvaluationRun, RunRecord, StopReason


def _sample_spec(kind: MemoryKind = MemoryKind.EPISODIC_STORE, persist: bool = True) -> AgentSpec:
    return AgentSpec(
        spec_id="test-spec",
        system_prompt="Test agent system prompt.",
        tools=("tool_a", "tool_b"),
        memory=MemoryConfig(
            kind=kind,
            retrieval_k=3 if kind in {MemoryKind.EPISODIC_STORE, MemoryKind.VECTOR_RETRIEVAL} else None,
            persist_across_runs=persist,
        ),
        stopping=StoppingConditions(max_steps=5),
    )


def test_episodic_entry_creation():
    entry = EpisodicEntry(
        entry_id="e1",
        kind="success",
        task_id="task_1",
        lesson="Solved with tool_a.",
        tools_used=("tool_a",),
        generation=1,
    )
    assert entry.entry_id == "e1"
    assert entry.kind == "success"
    assert entry.summary_line() == "[SUCCESS] Task task_1: Solved with tool_a."


def test_episodic_store_in_memory_and_clear():
    store = EpisodicMemoryStore()
    assert len(store) == 0

    entry1 = EpisodicEntry(
        entry_id="e1",
        kind="success",
        task_id="task_1",
        lesson="Look up tool worked.",
        tools_used=("lookup",),
        generation=0,
    )
    entry2 = EpisodicEntry(
        entry_id="e2",
        kind="failure_lesson",
        task_id="task_2",
        lesson="Calling calculate before lookup failed.",
        cause=FailureCause.TOOL_MISUSE,
        tools_used=("calculate",),
        generation=0,
    )

    store.add(entry1)
    store.add(entry2)
    assert len(store) == 2

    # Duplicate rejection
    store.add(entry1)
    assert len(store) == 2

    # Retrieval
    retrieved = store.retrieve("lookup", k=1)
    assert len(retrieved) == 1
    assert retrieved[0].entry_id == "e1"

    # Context formatting
    context = store.format_for_context(retrieved)
    assert "Episodic Memory" in context
    assert "Look up tool worked" in context

    # Clear
    store.clear()
    assert len(store) == 0


def test_episodic_store_file_persistence(tmp_path: Path):
    file_path = tmp_path / "memory.json"
    store1 = EpisodicMemoryStore(storage_path=file_path)
    entry = EpisodicEntry(
        entry_id="e_saved",
        kind="success",
        task_id="t_save",
        lesson="Persisted lesson",
        generation=1,
    )
    store1.add(entry)
    assert file_path.is_file()

    # Load from another instance
    store2 = EpisodicMemoryStore(storage_path=file_path)
    assert len(store2) == 1
    assert store2.entries[0].entry_id == "e_saved"
    assert store2.entries[0].lesson == "Persisted lesson"

    # Clear removes file and in-memory entries
    store2.clear()
    assert len(store2) == 0
    assert not file_path.exists()


def test_self_reflection_distills_success_and_failures():
    spec = _sample_spec()
    passed_traj = Trajectory(
        trajectory_id="t_pass",
        spec_id=spec.spec_id,
        task_id="task_pass",
        tool_calls=(
            ToolCall(ordinal=0, tool_name="tool_a", args={"x": 1}, result="ok_1"),
            ToolCall(ordinal=1, tool_name="tool_b", args={"y": 2}, result="ok_2"),
        ),
        final_answer="42",
        verdict=EvaluatorVerdict(evaluator_id="eval", passed=True, score=1.0),
    )
    failed_traj = Trajectory(
        trajectory_id="t_fail",
        spec_id=spec.spec_id,
        task_id="task_fail",
        tool_calls=(
            ToolCall(ordinal=0, tool_name="tool_a", args={"x": 1}, result="ok_1"),
            ToolCall(ordinal=1, tool_name="tool_b", args={"y": -1}, error="invalid negative arg"),
        ),
        final_answer=None,
        verdict=EvaluatorVerdict(evaluator_id="eval", passed=False, score=0.0, rationale="failed"),
    )

    run = EvaluationRun(
        spec_id=spec.spec_id,
        task_set_id="test_suite",
        records=(
            RunRecord(trajectory=passed_traj, stop_reason=StopReason.FINAL_ANSWER),
            RunRecord(trajectory=failed_traj, stop_reason=StopReason.STOP_ON_TOOL),
        ),
    )

    diagnosis = Diagnosis(
        diagnosis_id="diag_1",
        spec_id=spec.spec_id,
        attributions=(
            FailureAttribution(
                trajectory_id="t_fail",
                cause=FailureCause.TOOL_MISUSE,
                evidence="tool_b failed at step 1 with invalid negative arg",
                step_ordinal=1,
            ),
        ),
    )

    reflector = HeuristicReflector()
    entries = reflector.distill(spec, run, diagnosis, generation=1)

    assert len(entries) == 2
    success_entry = next(e for e in entries if e.kind == "success")
    failure_entry = next(e for e in entries if e.kind == "failure_lesson")

    assert success_entry.task_id == "task_pass"
    assert "tool_a -> tool_b" in success_entry.lesson

    assert failure_entry.task_id == "task_fail"
    assert failure_entry.cause == FailureCause.TOOL_MISUSE
    assert "tool_b" in failure_entry.lesson
    assert "tool_a" in failure_entry.lesson  # cites prior tool sequencing


def test_model_reflector_with_fallback():
    class DummyGenerator:
        def complete(self, system: str, prompt: str) -> str:
            return "Refined: verify tool arguments before invocation."

    spec = _sample_spec()
    failed_traj = Trajectory(
        trajectory_id="t_fail2",
        spec_id=spec.spec_id,
        task_id="task_fail2",
        final_answer=None,
        verdict=EvaluatorVerdict(evaluator_id="eval", passed=False, score=0.0),
    )
    run = EvaluationRun(
        spec_id=spec.spec_id,
        task_set_id="test_suite",
        records=(RunRecord(trajectory=failed_traj, stop_reason=StopReason.MAX_STEPS),),
    )
    diagnosis = Diagnosis(
        diagnosis_id="diag_2",
        spec_id=spec.spec_id,
        attributions=(
            FailureAttribution(
                trajectory_id="t_fail2",
                cause=FailureCause.STOPPING_CONDITION_HIT,
                evidence="max steps hit",
            ),
        ),
    )

    reflector = ModelReflector(DummyGenerator())
    entries = reflector.distill(spec, run, diagnosis, generation=1)
    assert len(entries) == 1
    assert "Refined: verify tool arguments before invocation." == entries[0].lesson

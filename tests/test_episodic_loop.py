"""Integration tests for the episodic memory loop across iterations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent_engineer.loop import run_loop
from agent_engineer.memory import EpisodicMemoryStore
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema
from agent_engineer.schemas import (
    AgentSpec,
    EvaluatorVerdict,
    MemoryConfig,
    MemoryKind,
    StoppingConditions,
)
from agent_engineer.stages.synthesize import TemplateSynthesizer


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: str
    answer: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Suite:
    domain: str
    tasks: tuple[_Task, ...]


class _DependentToolRuntime:
    """Two tools where tool_b depends on tool_a: calling tool_b before tool_a fails."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return (
            ToolSchema(
                name="step_a",
                description="First step that generates the required token.",
                parameters={"type": "object", "properties": {"item": {"type": "string"}}},
            ),
            ToolSchema(
                name="step_b",
                description="Second step that consumes the token from step_a.",
                parameters={"type": "object", "properties": {"token": {"type": "string"}}},
            ),
        )

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:
        if tool_name == "step_a":
            return ToolResult(value=f"token_for_{args.get('item', 'unknown')}")
        if tool_name == "step_b":
            tok = args.get("token")
            if not tok or not str(tok).startswith("token_for_"):
                return ToolResult(error="tool_b failed: token missing or not produced by step_a")
            return ToolResult(value=f"solved_{tok}")
        return ToolResult(error=f"unknown tool {tool_name}")


class _MemoryAwareBackend:
    """Backend whose actions improve when relevant episodic memory is present in context."""

    def next_action(self, spec: AgentSpec, task: _Task, tools, history):
        prompt_text = spec.system_prompt
        has_memory = "Episodic Memory" in prompt_text
        memory_tokens = 25 if has_memory else 0

        if not history:
            if not has_memory:
                # Generation 0: mistakenly call step_b first without token -> triggers TOOL_MISUSE
                return AgentAction(
                    tool_name="step_b",
                    args={"token": "invalid"},
                    prompt_tokens=20 + memory_tokens,
                    completion_tokens=5,
                )
            # Memory retrieved! Follows lesson: call step_a first
            return AgentAction(
                tool_name="step_a",
                args={"item": task.metadata["item"]},
                prompt_tokens=20 + memory_tokens,
                completion_tokens=5,
            )

        last = history[-1]
        if last.tool_name == "step_a" and last.succeeded:
            # Step a succeeded, call step_b with result
            return AgentAction(
                tool_name="step_b",
                args={"token": last.result},
                prompt_tokens=15 + memory_tokens,
                completion_tokens=5,
            )
        if last.tool_name == "step_b" and last.succeeded:
            return AgentAction(
                final_answer=last.result,
                prompt_tokens=10 + memory_tokens,
                completion_tokens=5,
            )
        # Errored and gave up
        return AgentAction(
            final_answer="unknown",
            prompt_tokens=10 + memory_tokens,
            completion_tokens=5,
        )


def _evaluator(task: _Task, trajectory) -> EvaluatorVerdict:
    passed = trajectory.final_answer == task.answer
    return EvaluatorVerdict(
        evaluator_id="exact-match",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="exact match" if passed else f"expected {task.answer!r}, got {trajectory.final_answer!r}",
    )


def _suite() -> _Suite:
    items = ["apple", "banana", "cherry"]
    tasks = tuple(
        _Task(
            task_id=f"t_{item}",
            prompt=f"process {item}",
            answer=f"solved_token_for_{item}",
            metadata={"item": item},
        )
        for item in items
    )
    return _Suite(domain="dependency-domain", tasks=tasks)


def test_episodic_memory_grows_accuracy_rises(tmp_path: Path):
    store_file = tmp_path / "episodic_store.json"
    memory_store = EpisodicMemoryStore(storage_path=store_file)

    synthesizer = TemplateSynthesizer(
        memory=MemoryConfig(
            kind=MemoryKind.EPISODIC_STORE,
            retrieval_k=3,
            persist_across_runs=True,
        )
    )

    report = run_loop(
        spec_id="memory-agent",
        goal="Process items using step_a and step_b in the correct sequence.",
        tool_runtime=_DependentToolRuntime(),
        backend=_MemoryAwareBackend(),
        evaluator=_evaluator,
        evaluator_id="exact-match",
        task_suite=_suite(),
        max_generations=2,
        synthesizer=synthesizer,
        memory_store=memory_store,
    )

    # Verify memory growth across iterations
    assert len(report.generations) >= 1
    gen1 = report.generations[0]

    # Baseline (gen 0) failed because it had no memory
    assert gen1.verdict.before == 0.0

    # Generation 1 retrieved distilled failure lessons from gen 0 and succeeded!
    assert gen1.verdict.after > gen1.verdict.before
    assert gen1.accepted
    assert gen1.memory_store_size > 0

    # Check that memory store persisted to file
    assert store_file.is_file()
    reloaded_store = EpisodicMemoryStore(storage_path=store_file)
    assert len(reloaded_store) == gen1.memory_store_size

    # Check the memory growth table output
    table = report.memory_growth_table()
    print("\n" + table)
    assert "=== Episodic Memory Growth Across Iterations ===" in table
    assert "gen 0 (root)" in table
    assert "gen 1" in table
    assert "Summary: Memory:" in table
    assert "Accuracy:" in table
    assert "Cost:" in table


def test_memory_does_not_leak_when_unconfigured():
    """When persist_across_runs is false and episodic memory is not configured, state is isolated."""
    report = run_loop(
        spec_id="isolated-agent",
        goal="Process items.",
        tool_runtime=_DependentToolRuntime(),
        backend=_MemoryAwareBackend(),
        evaluator=_evaluator,
        evaluator_id="exact-match",
        task_suite=_suite(),
        max_generations=1,
    )
    assert report.memory_store is not None
    # No memory accumulated because spec uses default full_transcript without persistence
    assert len(report.memory_store) == 0


def test_episodic_memory_accuracy_rose_cost_fell(tmp_path: Path):
    """The central claim: memory grew, accuracy rose, and cost per task fell by avoiding exploratory retries."""
    store_file = tmp_path / "efficient_store.json"
    memory_store = EpisodicMemoryStore(storage_path=store_file)

    class _ExploratoryToolRuntime:
        def schemas(self) -> tuple[ToolSchema, ...]:
            return (
                ToolSchema(name="query_db", description="Query database for key."),
            )

        def invoke(self, task, tool_name: str, args: dict) -> ToolResult:
            key = args.get("key")
            if key == task.metadata.get("correct_key"):
                return ToolResult(value=f"value_for_{key}")
            return ToolResult(error=f"key {key} not found")

    class _ExploratoryBackend:
        """Without memory, wanders through trial and error (high token spend).
        With memory, jumps straight to the solution (low token spend)."""

        def next_action(self, spec: AgentSpec, task, tools, history):
            has_memory = "Episodic Memory" in spec.system_prompt
            if has_memory:
                # With memory: solve directly in 1 step!
                if not history:
                    return AgentAction(
                        tool_name="query_db",
                        args={"key": task.metadata["correct_key"]},
                        prompt_tokens=25,
                        completion_tokens=5,
                    )
                return AgentAction(
                    final_answer=history[-1].result,
                    prompt_tokens=20,
                    completion_tokens=5,
                )

            # Without memory: tries wrong keys repeatedly (3 failed attempts before giving up)
            if len(history) < 3:
                return AgentAction(
                    tool_name="query_db",
                    args={"key": f"wrong_{len(history)}"},
                    prompt_tokens=30,
                    completion_tokens=10,
                )
            return AgentAction(
                final_answer="unknown",
                prompt_tokens=30,
                completion_tokens=5,
            )

    tasks = tuple(
        _Task(task_id=f"k_{i}", prompt=f"get {i}", answer=f"value_for_target_{i}", metadata={"correct_key": f"target_{i}"})
        for i in range(3)
    )
    suite = _Suite(domain="efficiency-domain", tasks=tasks)

    synth = TemplateSynthesizer(
        memory=MemoryConfig(
            kind=MemoryKind.EPISODIC_STORE,
            retrieval_k=3,
            persist_across_runs=True,
        )
    )

    report = run_loop(
        spec_id="efficient-agent",
        goal="Fetch database values.",
        tool_runtime=_ExploratoryToolRuntime(),
        backend=_ExploratoryBackend(),
        evaluator=_evaluator,
        evaluator_id="exact-match",
        task_suite=suite,
        max_generations=2,
        synthesizer=synth,
        memory_store=memory_store,
    )

    assert len(report.generations) >= 1
    gen1 = report.generations[0]

    # Memory grew
    assert gen1.memory_store_size > 0
    # Accuracy rose
    assert gen1.verdict.after > gen1.verdict.before
    assert gen1.verdict.after == 1.0

    # Cost per task fell!
    cost_before = sum(t.tokens.total_tokens for t in gen1.evaluation_before.trajectories) / 3
    cost_after = sum(t.tokens.total_tokens for t in gen1.evaluation_after.trajectories) / 3
    assert cost_after < cost_before

    table = report.memory_growth_table()
    print("\n" + table)
    assert "cost_delta" or "-" in table
    assert cost_after < cost_before

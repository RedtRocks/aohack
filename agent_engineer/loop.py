"""The control flow that drives the five stages end to end.

One call to :func:`run_loop` performs: synthesize once, then repeatedly
evaluate -> diagnose -> mutate -> re-evaluate -> keep-or-revert, for up to
``max_generations`` mutations. Nothing here is domain-specific: every argument
that could carry domain content -- the goal, the tools, the evaluator, the task
suite -- comes in from the caller, and this module never inspects their
content, only their shape.

The result is a :class:`LineageReport`: the sequence of specs actually kept,
and every :class:`~agent_engineer.stages.select.Verdict` produced along the
way, whether accepted or reverted. That is the artifact the integration gate
checks: a lineage of mutations, each carrying its measured delta and its
before number.

**The keep-or-revert threshold defaults to noise-aware, not a fixed epsilon.**
:class:`~agent_engineer.stages.select.MinimumDeltaPolicy`'s own default
(``min_delta=1e-9``) accepts any positive delta whatsoever, including one
smaller than the measurement noise in the metric itself -- and more retries
always buys accuracy, so a threshold that cannot tell a real improvement from
noise will happily accept neither. When the caller does not supply a
``selection_policy``, this module measures a real reliability variance for the
root spec through the frozen :class:`~agent_engineer.evaluation.Evaluator`
harness (``repeats=3``) before running any generation, and uses the resulting
standard deviation as the keep threshold for the whole lineage -- floored at
the same tiny epsilon so a fully deterministic configuration (zero measured
noise) still reverts a flat delta exactly as before. A caller who wants the
old, permissive behaviour has to ask for it explicitly, by passing their own
``selection_policy=MinimumDeltaPolicy(min_delta=...)``, which also skips this
measurement entirely -- no surprise extra calls for a caller who opts out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_engineer.evaluation import Evaluator
from agent_engineer.memory import EpisodicMemoryStore, HeuristicReflector, Reflector
from agent_engineer.ports import DomainSuite, ModelBackend, TaskEvaluator, ToolRuntime
from agent_engineer.schemas import AgentSpec, Diagnosis, MemoryKind, Mutation
from agent_engineer.stages.diagnose import Diagnoser, HeuristicDiagnoser
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner
from agent_engineer.stages.mutate import LadderMutator, Mutator
from agent_engineer.stages.select import MinimumDeltaPolicy, SelectionPolicy, Verdict
from agent_engineer.stages.synthesize import SpecSynthesizer, TemplateSynthesizer

DEFAULT_RELIABILITY_REPEATS = 3
"""Repeats used to measure the default noise floor. The Evaluator harness
requires at least this many for a defined reliability metric."""

MIN_KEEP_DELTA_FLOOR = 1e-9
"""The floor under the derived threshold: a fully deterministic configuration
measures zero noise, and a flat delta must still revert, exactly as under the
old fixed-epsilon default."""


def _measure_noise_floor(
    spec: AgentSpec,
    task_suite: DomainSuite,
    evaluator: TaskEvaluator,
    runner: TrajectoryRunner,
    repeats: int,
) -> float:
    """A real run-to-run noise floor for ``spec`` on ``task_suite``, measured
    through the frozen Evaluator harness -- never asserted by fiat. Returns the
    population-variance-of-pass-indicator, converted to a standard deviation,
    floored at :data:`MIN_KEEP_DELTA_FLOOR`.
    """
    harness = Evaluator([task_suite], evaluators={task_suite.domain: evaluator}, repeats=repeats)
    report = harness.run_iteration(0, spec, runner.as_task_runner())
    reliability = report.domain(task_suite.domain).reliability
    if not reliability.is_defined:
        return MIN_KEEP_DELTA_FLOOR
    return max(MIN_KEEP_DELTA_FLOOR, reliability.value**0.5)


@dataclass(frozen=True)
class GenerationRecord:
    """One mutation attempt: the proposal, what it measured, and the outcome."""

    generation: int
    mutation: Mutation
    diagnosis: Diagnosis
    evaluation_before: EvaluationRun
    evaluation_after: EvaluationRun
    verdict: Verdict
    spec_before: AgentSpec
    spec_after: AgentSpec
    memory_store_size: int = 0

    @property
    def accepted(self) -> bool:
        return self.verdict.accepted


@dataclass(frozen=True)
class LineageReport:
    """The full record of one run of the loop: every spec kept, every mutation tried."""

    root_spec: AgentSpec
    generations: tuple[GenerationRecord, ...]
    noise_floor: float | None = None
    memory_store: EpisodicMemoryStore | None = None
    """The keep threshold actually used, when derived automatically from a measured
    reliability variance (see the module docstring). ``None`` when the caller
    supplied their own ``selection_policy`` -- the derivation never ran, so there
    is no measured floor to report."""

    @property
    def final_spec(self) -> AgentSpec:
        """The spec actually in force after the loop: the last accepted mutation's child,
        or the root if none were accepted."""
        for record in reversed(self.generations):
            if record.accepted:
                return record.spec_after
        return self.root_spec

    @property
    def accepted_mutations(self) -> tuple[Mutation, ...]:
        return tuple(record.mutation for record in self.generations if record.accepted)

    @property
    def mutations(self) -> tuple[Mutation, ...]:
        return tuple(record.mutation for record in self.generations)

    def summary_lines(self) -> tuple[str, ...]:
        """One line per generation: cause targeted, decision, and the numbers behind it."""
        lines = []
        for record in self.generations:
            outcome = "accepted" if record.accepted else "reverted"
            lines.append(
                f"gen {record.generation}: {record.mutation.motivating_cause.value} -> "
                f"{record.mutation.kind.value} ({record.mutation.target_path}) | "
                f"{record.verdict.before:.4f} -> {record.verdict.after:.4f} "
                f"(delta {record.verdict.delta:+.4f}) | {outcome}"
            )
        return tuple(lines)

    def memory_growth_table(self) -> str:
        """Render a table across iterations showing memory store size, accuracy, and cost per task."""
        if not self.generations:
            return "No generations recorded."

        gen0_run = self.generations[0].evaluation_before
        gen0_trajs = gen0_run.trajectories
        gen0_cost = (
            sum(t.tokens.total_tokens for t in gen0_trajs) / max(1, len(gen0_trajs))
        )
        gen0_acc = gen0_run.pass_rate

        headers = ["Iteration", "Memory Size", "Accuracy", "Cost / Task", "Verdict"]
        rows: list[list[str]] = [
            ["gen 0 (root)", "0 entries", f"{gen0_acc:.4f}", f"{gen0_cost:.1f} tok", "baseline"]
        ]

        for record in self.generations:
            after_run = record.evaluation_after
            after_trajs = after_run.trajectories
            after_cost = (
                sum(t.tokens.total_tokens for t in after_trajs) / max(1, len(after_trajs))
            )
            after_acc = after_run.pass_rate
            decision = "ACCEPTED" if record.accepted else "REVERTED"
            acc_delta = after_acc - record.verdict.before
            cost_delta = after_cost - gen0_cost
            rows.append(
                [
                    f"gen {record.generation}",
                    f"{record.memory_store_size} entries",
                    f"{after_acc:.4f} ({acc_delta:+.4f})",
                    f"{after_cost:.1f} tok ({cost_delta:+.1f} tok)",
                    decision,
                ]
            )

        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        lines = [
            "=== Episodic Memory Growth Across Iterations ===",
            "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
            "  ".join("-" * widths[i] for i in range(len(headers))),
        ]
        for row in rows:
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

        last_rec = self.generations[-1]
        last_trajs = last_rec.evaluation_after.trajectories
        last_cost = (
            sum(t.tokens.total_tokens for t in last_trajs) / max(1, len(last_trajs))
        )
        last_acc = last_rec.evaluation_after.pass_rate
        total_acc_delta = last_acc - gen0_acc
        total_cost_delta = last_cost - gen0_cost
        lines.append("-------------------------------------------------")
        lines.append(
            f"Summary: Memory: 0 -> {last_rec.memory_store_size} entries (+{last_rec.memory_store_size}) | "
            f"Accuracy: {gen0_acc:.4f} -> {last_acc:.4f} ({total_acc_delta:+.4f}) | "
            f"Cost: {gen0_cost:.1f} tok -> {last_cost:.1f} tok ({total_cost_delta:+.1f} tok)"
        )
        return "\n".join(lines)


class LoopStalled(RuntimeError):
    """Raised when the mutator has no move left to propose for the dominant cause."""


def run_loop(
    *,
    spec_id: str,
    goal: str,
    tool_runtime: ToolRuntime,
    backend: ModelBackend,
    evaluator: TaskEvaluator,
    evaluator_id: str,
    task_suite: DomainSuite,
    max_generations: int = 3,
    evaluator_criteria: str = "",
    synthesizer: SpecSynthesizer | None = None,
    diagnoser: Diagnoser | None = None,
    mutator: Mutator | None = None,
    selection_policy: SelectionPolicy | None = None,
    metric: str = "pass_rate",
    stop_on_stall: bool = True,
    on_generation: Callable[[GenerationRecord], None] | None = None,
    reliability_repeats: int = DEFAULT_RELIABILITY_REPEATS,
    memory_store: EpisodicMemoryStore | None = None,
    memory_store_path: str | Path | None = None,
    reflector: Reflector | None = None,
) -> LineageReport:
    """Run the full agent-engineer loop for one domain and return its lineage.

    ``metric`` selects which scalar on :class:`~agent_engineer.stages.evaluate.EvaluationRun`
    stage 5 compares (``"pass_rate"`` or ``"mean_score"``); either is derived
    purely from the verdicts the supplied evaluator already produced, never
    recomputed by the engine.

    ``on_generation``, if given, is called once per generation with the
    :class:`GenerationRecord` just produced, in order, before the loop moves on
    to the next one. It exists so a caller (a CLI, a progress bar) can stream
    the lineage as it is built instead of waiting for the whole
    :class:`LineageReport`; the loop's own control flow and return value are
    unaffected by whether one is supplied.

    When ``selection_policy`` is not supplied, this measures the root spec's
    reliability through the Evaluator harness (``repeats=reliability_repeats``,
    at least 3) before running any generation, and keeps only deltas that clear
    that measured noise floor -- see the module docstring. Passing an explicit
    ``selection_policy`` skips this measurement entirely.
    """
    synthesizer = synthesizer or TemplateSynthesizer()
    diagnoser = diagnoser or HeuristicDiagnoser()
    mutator = mutator or LadderMutator()
    reflector = reflector or HeuristicReflector()
    if memory_store is None:
        memory_store = EpisodicMemoryStore(storage_path=memory_store_path)

    runner = TrajectoryRunner(
        backend=backend, tool_runtime=tool_runtime, evaluator=evaluator, memory_store=memory_store
    )

    root_spec = synthesizer.synthesize(
        spec_id=spec_id,
        goal=goal,
        tools=tool_runtime.schemas(),
        evaluator_id=evaluator_id,
        criteria=evaluator_criteria,
    )

    noise_floor: float | None = None
    if selection_policy is None:
        noise_floor = _measure_noise_floor(
            root_spec, task_suite, evaluator, runner, reliability_repeats
        )
        selection_policy = MinimumDeltaPolicy(min_delta=noise_floor)
        memory_store.clear()

    def score(run: EvaluationRun) -> float:
        return run.pass_rate if metric == "pass_rate" else run.mean_score

    current_spec = root_spec
    current_run = runner.run_suite(current_spec, task_suite, generation=0)
    generations: list[GenerationRecord] = []
    already_tried: set[tuple[str, str]] = set()

    for generation in range(1, max_generations + 1):
        diagnosis = diagnoser.diagnose(
            current_spec, current_run, diagnosis_id=f"{current_spec.spec_id}::diag{generation}"
        )
        if diagnosis.dominant_cause is None:
            break  # nothing failed; there is nothing left to fix

        # Self-reflection: if current_spec has episodic memory active, distill lessons
        if current_spec.memory.kind is MemoryKind.EPISODIC_STORE and current_spec.memory.persist_across_runs:
            reflections = reflector.distill(
                current_spec, current_run, diagnosis, generation=generation - 1
            )
            memory_store.add_many(reflections)

        proposal = mutator.propose_with_child(
            current_spec,
            diagnosis,
            mutation_id=f"{current_spec.spec_id}::mut{generation}",
            child_spec_id=f"{spec_id}-g{generation}",
            already_tried=frozenset(already_tried),
        )
        if proposal is None:
            if stop_on_stall:
                break
            raise LoopStalled(
                f"generation {generation}: no move left for cause {diagnosis.dominant_cause.value}"
            )
        mutation, child_spec = proposal
        already_tried.add((mutation.kind.value, mutation.target_path))

        # If child_spec introduced episodic memory, distill lessons from prior run now
        if (
            child_spec.memory.kind is MemoryKind.EPISODIC_STORE
            and child_spec.memory.persist_across_runs
            and len(memory_store) == 0
        ):
            reflections = reflector.distill(
                current_spec, current_run, diagnosis, generation=generation - 1
            )
            memory_store.add_many(reflections)

        child_run = runner.run_suite(child_spec, task_suite, generation=generation)
        verdict = selection_policy.decide(before=score(current_run), after=score(child_run))

        # Accumulate reflections from child_run when child_spec has episodic memory
        if child_spec.memory.kind is MemoryKind.EPISODIC_STORE and child_spec.memory.persist_across_runs:
            child_diag = diagnoser.diagnose(
                child_spec, child_run, diagnosis_id=f"{child_spec.spec_id}::diag{generation}"
            )
            child_reflections = reflector.distill(
                child_spec, child_run, child_diag, generation=generation
            )
            memory_store.add_many(child_reflections)

        record = GenerationRecord(
            generation=generation,
            mutation=mutation,
            diagnosis=diagnosis,
            evaluation_before=current_run,
            evaluation_after=child_run,
            verdict=verdict,
            spec_before=current_spec,
            spec_after=child_spec,
            memory_store_size=len(memory_store),
        )
        generations.append(record)
        if on_generation is not None:
            on_generation(record)

        if verdict.accepted:
            current_spec = child_spec
            current_run = child_run
        # reverted: current_spec/current_run stay put, but already_tried keeps the
        # ladder moving forward on the next generation instead of re-proposing this edit

    return LineageReport(
        root_spec=root_spec,
        generations=tuple(generations),
        noise_floor=noise_floor,
        memory_store=memory_store,
    )

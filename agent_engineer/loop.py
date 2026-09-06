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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_engineer.ports import DomainSuite, ModelBackend, TaskEvaluator, ToolRuntime
from agent_engineer.schemas import AgentSpec, Diagnosis, Mutation
from agent_engineer.stages.diagnose import Diagnoser, HeuristicDiagnoser
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner
from agent_engineer.stages.mutate import LadderMutator, Mutator
from agent_engineer.stages.select import MinimumDeltaPolicy, SelectionPolicy, Verdict
from agent_engineer.stages.synthesize import SpecSynthesizer, TemplateSynthesizer


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

    @property
    def accepted(self) -> bool:
        return self.verdict.accepted


@dataclass(frozen=True)
class LineageReport:
    """The full record of one run of the loop: every spec kept, every mutation tried."""

    root_spec: AgentSpec
    generations: tuple[GenerationRecord, ...]

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
    """
    synthesizer = synthesizer or TemplateSynthesizer()
    diagnoser = diagnoser or HeuristicDiagnoser()
    mutator = mutator or LadderMutator()
    selection_policy = selection_policy or MinimumDeltaPolicy()
    runner = TrajectoryRunner(backend=backend, tool_runtime=tool_runtime, evaluator=evaluator)

    root_spec = synthesizer.synthesize(
        spec_id=spec_id,
        goal=goal,
        tools=tool_runtime.schemas(),
        evaluator_id=evaluator_id,
        criteria=evaluator_criteria,
    )

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

        child_run = runner.run_suite(child_spec, task_suite, generation=generation)
        verdict = selection_policy.decide(before=score(current_run), after=score(child_run))

        record = GenerationRecord(
            generation=generation,
            mutation=mutation,
            diagnosis=diagnosis,
            evaluation_before=current_run,
            evaluation_after=child_run,
            verdict=verdict,
            spec_before=current_spec,
            spec_after=child_spec,
        )
        generations.append(record)
        if on_generation is not None:
            on_generation(record)

        if verdict.accepted:
            current_spec = child_spec
            current_run = child_run
        # reverted: current_spec/current_run stay put, but already_tried keeps the
        # ladder moving forward on the next generation instead of re-proposing this edit

    return LineageReport(root_spec=root_spec, generations=tuple(generations))

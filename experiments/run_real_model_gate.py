"""Run the real five-stage loop against real GPT-5 nano on code_math, and save the
verbatim lineage plus a harness-computed accuracy/cost/reliability comparison.

This answers the question the scripted gate test (``tests/test_code_math_loop_end_to_end.py``)
cannot: does the engine's mutate/select machinery actually improve a real agent,
not just wire together correctly against a stand-in whose competence was a
function of the generation number. Nothing here is a domain module and nothing
here is imported by the engine; it is a one-off experiment that produces a
repo artifact.

What it does, in order, all against the real ``agent_engineer.domains.code_math``
suite and evaluator, with :class:`~agent_engineer.ports.ModelBackend` supplied
by real GPT-5 nano (``experiments/gpt5_nano_backend.py``):

1. Synthesizes one root :class:`AgentSpec` (via the engine's own
   ``TemplateSynthesizer`` -- generic scaffolding, no domain vocabulary).
2. Runs that root spec through the real suite three independent times, to
   measure the run-to-run noise of ``mean_score`` -- the very metric
   :func:`agent_engineer.loop.run_loop` uses to accept or revert a mutation.
   Any lineage delta smaller than this noise is flagged, not reported as an
   improvement.
3. Feeds those same three runs into the real, frozen
   :class:`agent_engineer.evaluation.Evaluator` (``repeats=3``) to get the
   canonical accuracy/cost/reliability/speed numbers for the baseline spec --
   accuracy is never reported here without its paired cost.
4. Runs :func:`agent_engineer.loop.run_loop`, unmodified, for real generations
   of mutation -> measure -> accept-or-revert.
5. Repeats step 2-3 for the loop's final spec, so baseline and final are
   measured the same way and are directly comparable.

Requires ``OPENAI_API_KEY`` in the environment. Stops immediately, without
simulating anything, if it is not set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt5_nano_backend import ApiKeyMissing, GPT5NanoBackend, MODEL  # noqa: E402

from agent_engineer.domains.code_math import DOMAIN, EVALUATOR_ID, get_evaluator, get_suite  # noqa: E402
from agent_engineer.evaluation import Evaluator  # noqa: E402
from agent_engineer.evaluation.metrics import population_variance  # noqa: E402
from agent_engineer.loop import run_loop  # noqa: E402
from agent_engineer.ports import ToolResult, ToolSchema  # noqa: E402
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner  # noqa: E402
from agent_engineer.stages.synthesize import TemplateSynthesizer  # noqa: E402

SPEC_ID = "code-math-gpt5-nano"
GOAL = "Solve each task exactly, in the exact output format the prompt asks for."
MAX_GENERATIONS = 4
NOISE_REPLICATES = 3
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class _NoTools:
    """code_math needs no tools: every task is answered directly."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


class _FixedSpecSynthesizer:
    """Hands back one pre-built spec, so the loop's root is identical to the
    spec the noise-floor and baseline-harness measurements were taken on."""

    def __init__(self, spec) -> None:
        self._spec = spec

    def synthesize(self, **kwargs):
        return self._spec


def _replicate_runs(runner: TrajectoryRunner, suite, spec, *, n: int, tag: str) -> list[EvaluationRun]:
    return [runner.run_suite(spec, suite, generation=f"{tag}{i}") for i in range(n)]


def _replay_runner(replicates: list[EvaluationRun]):
    """A TaskRunner that replays already-collected real trajectories instead of
    calling the model again, so the harness's canonical accuracy/cost/reliability
    numbers are computed over the exact same real runs the noise floor used."""
    lookup = {}
    for attempt, run in enumerate(replicates):
        for record in run.records:
            lookup[(record.trajectory.task_id, attempt)] = record.trajectory

    def replay(spec, task, attempt):
        return lookup[(task.task_id, attempt)]

    return replay


def _metric_dict(metric) -> dict:
    return {"value": metric.value, "sample_size": metric.sample_size, "reason": metric.reason}


def _domain_report_dict(iteration_report, domain: str) -> dict:
    report = iteration_report.domain(domain)
    return {
        "accuracy": _metric_dict(report.accuracy),
        "reliability": _metric_dict(report.reliability),
        "cost_tokens": _metric_dict(report.cost),
        "speed_seconds": _metric_dict(report.speed),
        "task_count": report.task_count,
        "repeats": report.repeats,
    }


def main() -> None:
    try:
        backend = GPT5NanoBackend()
    except ApiKeyMissing as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(1)

    tool_runtime = _NoTools()
    suite = get_suite()
    evaluator = get_evaluator()
    runner = TrajectoryRunner(backend=backend, tool_runtime=tool_runtime, evaluator=evaluator)
    harness = Evaluator([suite], evaluators={DOMAIN: evaluator}, repeats=NOISE_REPLICATES)

    root_spec = TemplateSynthesizer().synthesize(
        spec_id=SPEC_ID, goal=GOAL, tools=tool_runtime.schemas(), evaluator_id=EVALUATOR_ID
    )

    print(f"[1/5] measuring baseline noise floor: {NOISE_REPLICATES} real replicate suite runs "
          f"on the root spec ({len(suite.tasks)} tasks each)...", flush=True)
    baseline_replicates = _replicate_runs(runner, suite, root_spec, n=NOISE_REPLICATES, tag="baseline")
    baseline_scores = [run.mean_score for run in baseline_replicates]
    noise_variance = population_variance(baseline_scores)
    print(f"    baseline mean_score across replicates: {baseline_scores}", flush=True)

    print("[2/5] feeding those same real runs into the frozen Evaluator harness "
          "for canonical accuracy+cost+reliability...", flush=True)
    baseline_report = harness.run_iteration(0, root_spec, _replay_runner(baseline_replicates))

    print(f"[3/5] running the real five-stage loop ({MAX_GENERATIONS} generations max)...", flush=True)
    lineage = run_loop(
        spec_id=SPEC_ID,
        goal=GOAL,
        tool_runtime=tool_runtime,
        backend=backend,
        evaluator=evaluator,
        evaluator_id=EVALUATOR_ID,
        task_suite=suite,
        max_generations=MAX_GENERATIONS,
        metric="mean_score",
        synthesizer=_FixedSpecSynthesizer(root_spec),
    )
    for line in lineage.summary_lines():
        print(f"    {line}", flush=True)

    if lineage.final_spec.spec_id != root_spec.spec_id:
        print("[4/5] measuring the final spec the same way as the baseline...", flush=True)
        final_replicates = _replicate_runs(
            runner, suite, lineage.final_spec, n=NOISE_REPLICATES, tag="final"
        )
        final_report = harness.run_iteration(1, lineage.final_spec, _replay_runner(final_replicates))
    else:
        print("[4/5] no mutation was ever accepted; final spec == root spec, reusing baseline.",
              flush=True)
        final_replicates = baseline_replicates
        final_report = baseline_report

    print(f"[5/5] writing artifact ({backend.calls_made} real model calls made total)...", flush=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": MODEL,
        "domain": DOMAIN,
        "spec_id": SPEC_ID,
        "goal": GOAL,
        "task_count": len(suite.tasks),
        "real_model_calls_made": backend.calls_made,
        "noise_floor": {
            "metric": "mean_score",
            "replicate_scores": baseline_scores,
            "population_variance": noise_variance,
            "population_std": noise_variance**0.5,
            "replicates": NOISE_REPLICATES,
        },
        "baseline": _domain_report_dict(baseline_report, DOMAIN),
        "final": _domain_report_dict(final_report, DOMAIN),
        "lineage": [
            {
                "generation": record.generation,
                "motivating_cause": record.mutation.motivating_cause.value,
                "mutation_kind": record.mutation.kind.value,
                "target_path": record.mutation.target_path,
                "before": record.verdict.before,
                "after": record.verdict.after,
                "delta": record.verdict.delta,
                "decision": record.verdict.decision.value,
                "within_noise_floor": abs(record.verdict.delta) < noise_variance**0.5,
                "rationale": record.mutation.rationale,
            }
            for record in lineage.generations
        ],
        "root_spec_id": lineage.root_spec.spec_id,
        "final_spec_id": lineage.final_spec.spec_id,
    }
    (ARTIFACT_DIR / "code_math_gpt5_nano_lineage.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print("done.", flush=True)


if __name__ == "__main__":
    main()

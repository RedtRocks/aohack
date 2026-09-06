"""Run the real five-stage loop against a real model on code_math, and save the
verbatim lineage plus a harness-computed accuracy/cost/reliability comparison.

This answers the question the scripted gate test (``tests/test_code_math_loop_end_to_end.py``)
cannot: does the engine's mutate/select machinery actually improve a real agent,
not just wire together correctly against a stand-in whose competence was a
function of the generation number. Nothing here is a domain module and nothing
here is imported by the engine; it is a one-off experiment that produces a
repo artifact.

What it does, in order, all against the real ``agent_engineer.domains.code_math``
suite and evaluator, with :class:`~agent_engineer.ports.ModelBackend` supplied
by a real model (``experiments/gpt5_nano_backend.py``): GLM 4.7 Flash via
TensorMux is the primary, GPT-5 nano is the fallback if the primary fails or
rate-limits (see :class:`~gpt5_nano_backend.FallbackBackend`).

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

Requires ``TENSORMUX_API_KEY`` (primary) and ``OPENAI_API_KEY`` (fallback) in
the environment. Stops immediately, without simulating anything, if either is
not set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt5_nano_backend import (  # noqa: E402
    ApiKeyMissing,
    CachingBackend,
    FallbackBackend,
    GPT5NanoBackend,
    MeteringBackend,
    SpendCapExceeded,
    TensorMuxGLMBackend,
)

import agent_engineer.domains as domain_registry  # noqa: E402
from agent_engineer.domains.code_math import EVALUATOR_ID as CM_EVALUATOR_ID  # noqa: E402
from agent_engineer.domains.api_orchestration.evaluator import EVALUATOR_ID as AO_EVALUATOR_ID  # noqa: E402
from api_orchestration_runtime import ApiOrchestrationToolRuntime  # noqa: E402
from agent_engineer.domains.extraction.evaluator import EVALUATOR_ID as EX_EVALUATOR_ID  # noqa: E402
from agent_engineer.evaluation import DomainSuite, Evaluator  # noqa: E402
from agent_engineer.evaluation.metrics import population_variance  # noqa: E402
from agent_engineer.loop import run_loop  # noqa: E402
from agent_engineer.ports import ToolResult, ToolSchema  # noqa: E402
from agent_engineer.schemas import (  # noqa: E402
    AgentSpec,
    MemoryConfig,
    MemoryKind,
    OrchestrationStrategy,
    StoppingConditions,
)
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner  # noqa: E402
from agent_engineer.stages.select import MinimumDeltaPolicy  # noqa: E402
from agent_engineer.stages.synthesize import TemplateSynthesizer  # noqa: E402

EVALUATOR_IDS = {
    "code_math": CM_EVALUATOR_ID,
    "api_orchestration": AO_EVALUATOR_ID,
    "extraction": EX_EVALUATOR_ID,
}

SPEC_ID = "code-math-gpt5-nano"
GOAL = "Solve each task exactly, in the exact output format the prompt asks for."
MAX_GENERATIONS = 4
NOISE_REPLICATES = 3
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CACHE_PATH = ARTIFACT_DIR / "real_model_call_cache.json"

DEFAULT_SPEND_CAP_TOKENS = 300_000
"""Hard cap on cumulative tokens for one run of this script. Overridable via
the SPEND_CAP_TOKENS env var. Crossing it raises SpendCapExceeded immediately
-- mid-phase, not just discovered after the fact -- because the keys this
script spends against are limited and there is no second set."""


class _NoTools:
    """Every task is answered directly: no tools exposed to the model."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


NAIVE_BASELINE_STATEMENT = (
    "We start from a naive baseline so that improvement is measurable. "
    "Generation zero is a bare system prompt (the goal string, verbatim, and "
    "nothing else -- no tool listing, no worked examples, no 'how to work' "
    "scaffolding), single_shot strategy, memory kind none, and a minimal "
    "one-step stopping condition. This is a deliberate experimental choice, "
    "not the strongest agent we could build: TemplateSynthesizer's own default "
    "prompt already tells the model to check tool errors and stop only once "
    "complete, which leaves little headroom for the loop's early mutations to "
    "visibly improve on. This baseline is weaker on purpose."
)


class _NaiveSynthesizer:
    """The deliberately weakest defensible starting agent.

    See :data:`NAIVE_BASELINE_STATEMENT` -- this exists so that generation
    zero has real room for the loop to gain, and its use is declared in the
    artifact and in this module rather than silently substituted for the
    engine's own default.
    """

    def synthesize(self, *, spec_id, goal, tools, evaluator_id, criteria=""):
        del tools, evaluator_id, criteria  # the naive prompt names none of this
        if not goal.strip():
            raise ValueError("goal must not be blank")
        return AgentSpec(
            spec_id=spec_id,
            system_prompt=goal.strip(),
            strategy=OrchestrationStrategy.SINGLE_SHOT,
            memory=MemoryConfig(kind=MemoryKind.NONE),
            stopping=StoppingConditions(max_steps=1),
        )


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
    parser = argparse.ArgumentParser(description="Run real model gate on a domain.")
    parser.add_argument(
        "--domain",
        default="code_math",
        choices=["code_math", "api_orchestration", "extraction"],
        help="Domain to run",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (e.g. 3 for pilot)",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=MAX_GENERATIONS,
        help="Max generations in loop",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=NOISE_REPLICATES,
        help="Number of replicates for noise floor",
    )
    parser.add_argument(
        "--spec-id",
        default=None,
        help="Custom spec ID (defaults to <domain>-gpt5-nano)",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Custom artifact filename (defaults to <domain>_gpt5_nano_lineage.json)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 3-task pilot with repeats=3, 1 generation and verify tokens/elapsed_seconds",
    )
    parser.add_argument(
        "--baseline",
        choices=["template", "naive"],
        default="template",
        help=(
            "Generation-zero synthesizer. 'template' (default) is the engine's own "
            "TemplateSynthesizer -- a competent starting agent. 'naive' is the "
            "deliberately weakest defensible starting agent (see "
            "NAIVE_BASELINE_STATEMENT): a bare system prompt, single_shot strategy, "
            "no memory, one-step stopping. Declared explicitly because it changes "
            "what 'improvement' means for the resulting lineage."
        ),
    )
    args = parser.parse_args()

    domain = args.domain
    max_generations = 1 if args.pilot else args.max_generations
    repeats = args.repeats
    max_tasks = 3 if args.pilot else args.max_tasks
    spec_id = args.spec_id or ("code-math-pilot" if args.pilot else (SPEC_ID if domain == "code_math" else f"{domain.replace('_', '-')}-gpt5-nano"))
    artifact_name = args.artifact_name or ("code_math_pilot_lineage.json" if args.pilot else f"{domain}_gpt5_nano_lineage.json")

    try:
        primary = TensorMuxGLMBackend()
        secondary = GPT5NanoBackend()
    except ApiKeyMissing as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(1)

    spend_cap = int(os.environ.get("SPEND_CAP_TOKENS", DEFAULT_SPEND_CAP_TOKENS))
    print(f"domain: {domain}, spec_id: {spec_id}, artifact: {artifact_name}", flush=True)
    print(f"spend cap for this run: {spend_cap} tokens", flush=True)
    # every call, real or replayed from cache, is logged and checked against the
    # cap the instant it happens -- not once per phase, so a kill mid-run leaves
    # an exact, not estimated, spend figure behind
    fallback = FallbackBackend(primary, secondary)
    backend = MeteringBackend(
        CachingBackend(fallback, cache_path=CACHE_PATH),
        max_total_tokens=spend_cap,
    )

    full_suite = domain_registry.get_suite(domain)
    if max_tasks is not None:
        suite = DomainSuite(domain=full_suite.domain, tasks=full_suite.tasks[:max_tasks])
    else:
        suite = full_suite
    # api_orchestration's tasks are only solvable through the tools in their own
    # metadata (per tasks.py); ApiOrchestrationToolRuntime is that domain's own
    # ToolRuntime adapter, wired in the same shape mcp_everything's MCPToolRuntime
    # would be. Every other domain here answers directly, no tools exposed.
    if domain == "api_orchestration":
        tool_runtime = ApiOrchestrationToolRuntime(suite.tasks)
    else:
        tool_runtime = _NoTools()
    evaluator = domain_registry.get_evaluator(domain)
    evaluator_id = EVALUATOR_IDS[domain]
    runner = TrajectoryRunner(backend=backend, tool_runtime=tool_runtime, evaluator=evaluator)
    harness = Evaluator([suite], evaluators={domain: evaluator}, repeats=repeats)

    if args.baseline == "naive":
        print(f"NAIVE BASELINE: {NAIVE_BASELINE_STATEMENT}", flush=True)
        synthesizer = _NaiveSynthesizer()
    else:
        synthesizer = TemplateSynthesizer()
    root_spec = synthesizer.synthesize(
        spec_id=spec_id, goal=GOAL, tools=tool_runtime.schemas(), evaluator_id=evaluator_id
    )

    def _write_truncated(reason: str) -> None:
        """A partial, honestly-labelled result is a perfectly good outcome when
        the spend cap fires -- crashing with a traceback and nothing saved is not."""
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        partial = {
            "status": "truncated",
            "reason": reason,
            "real_model_calls_made": backend.calls_made,
            "cumulative_tokens": backend.total_tokens,
            "primary_calls": fallback.primary_calls,
            "fallback_calls": fallback.secondary_calls,
        }
        (ARTIFACT_DIR / artifact_name).write_text(
            json.dumps(partial, indent=2), encoding="utf-8"
        )
        print(f"TRUNCATED: {reason}", file=sys.stderr)
        print(f"  calls made: {backend.calls_made} ({fallback.primary_calls} primary / "
              f"{fallback.secondary_calls} fallback), cumulative tokens: {backend.total_tokens}",
              file=sys.stderr)

    try:
        print(f"[1/5] measuring baseline noise floor: {repeats} real replicate suite runs "
              f"on the root spec ({len(suite.tasks)} tasks each)...", flush=True)
        baseline_replicates = _replicate_runs(runner, suite, root_spec, n=repeats, tag="baseline")
        baseline_scores = [run.mean_score for run in baseline_replicates]
        noise_variance = population_variance(baseline_scores)
        print(f"    baseline mean_score across replicates: {baseline_scores}", flush=True)
        print(f"    cumulative spend so far: {backend.total_tokens} tokens", flush=True)

        print("[2/5] feeding those same real runs into the frozen Evaluator harness "
              "for canonical accuracy+cost+reliability...", flush=True)
        baseline_report = harness.run_iteration(0, root_spec, _replay_runner(baseline_replicates))

        print(f"[3/5] running the real five-stage loop ({max_generations} generations max)...", flush=True)
        # run_loop's own default would otherwise measure this same noise floor itself
        # (another real repeats=3 pass, another len(suite.tasks)*3 calls) -- pass the
        # floor already measured above explicitly so that measurement is not paid for
        # twice.
        noise_floor_std = max(1e-9, noise_variance**0.5)
        lineage = run_loop(
            spec_id=spec_id,
            goal=GOAL,
            tool_runtime=tool_runtime,
            backend=backend,
            evaluator=evaluator,
            evaluator_id=evaluator_id,
            task_suite=suite,
            max_generations=max_generations,
            metric="mean_score",
            synthesizer=_FixedSpecSynthesizer(root_spec),
            selection_policy=MinimumDeltaPolicy(min_delta=noise_floor_std),
        )
        for line in lineage.summary_lines():
            print(f"    {line}", flush=True)
        print(f"    cumulative spend so far: {backend.total_tokens} tokens", flush=True)

        if lineage.final_spec.spec_id != root_spec.spec_id:
            print("[4/5] measuring the final spec the same way as the baseline...", flush=True)
            final_replicates = _replicate_runs(
                runner, suite, lineage.final_spec, n=repeats, tag="final"
            )
            final_report = harness.run_iteration(1, lineage.final_spec, _replay_runner(final_replicates))
        else:
            print("[4/5] no mutation was ever accepted; final spec == root spec, reusing baseline.",
                  flush=True)
            final_replicates = baseline_replicates
            final_report = baseline_report
    except SpendCapExceeded as error:
        _write_truncated(str(error))
        raise SystemExit(1)

    print(f"[5/5] writing artifact ({backend.calls_made} real model calls made total: "
          f"{fallback.primary_calls} primary / {fallback.secondary_calls} fallback, "
          f"{backend.total_tokens} cumulative tokens)...", flush=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "primary_model": "glm-4-7-flash (via TensorMux)",
        "fallback_model": "gpt-5-nano (via OpenAI)",
        "primary_calls": fallback.primary_calls,
        "fallback_calls": fallback.secondary_calls,
        "domain": domain,
        "spec_id": spec_id,
        "goal": GOAL,
        "baseline_synthesizer": args.baseline,
        "baseline_statement": (
            NAIVE_BASELINE_STATEMENT if args.baseline == "naive" else
            "Generation zero used TemplateSynthesizer's competent default prompt, "
            "not a naive baseline."
        ),
        "task_count": len(suite.tasks),
        "real_model_calls_made": backend.calls_made,
        "cumulative_tokens": backend.total_tokens,
        "noise_floor": {
            "metric": "mean_score",
            "replicate_scores": baseline_scores,
            "population_variance": noise_variance,
            "population_std": noise_variance**0.5,
            "replicates": repeats,
        },
        "baseline": _domain_report_dict(baseline_report, domain),
        "final": _domain_report_dict(final_report, domain),
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
    (ARTIFACT_DIR / artifact_name).write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_DIR / artifact_name}", flush=True)

    try:
        from render_lineage_markdown import render
        md_name = Path(artifact_name).with_suffix(".md").name
        (ARTIFACT_DIR / md_name).write_text(render(artifact), encoding="utf-8")
        print(f"wrote {ARTIFACT_DIR / md_name}", flush=True)
    except Exception as exc:
        print(f"markdown rendering failed: {exc}", file=sys.stderr)

    if args.pilot:
        print("\n=== PILOT RUN VERIFICATION ===")
        print(f"Tasks: {len(suite.tasks)}, Repeats: {repeats}, Generations: {len(lineage.generations)}")
        print(f"Total model calls made: {backend.calls_made}")
        print(f"Total tokens spent: {backend.total_tokens}")
        print(f"Primary calls (TensorMux): {fallback.primary_calls}, Fallback calls: {fallback.secondary_calls}")
        assert backend.calls_made > 0, "No model calls were made!"
        assert backend.total_tokens > 0, "Zero tokens spent! Calls must report real cost."
        all_trajectories = [r.trajectory for run in baseline_replicates for r in run.records]
        for gen in lineage.generations:
            all_trajectories.extend([r.trajectory for r in gen.evaluation_before.records])
            all_trajectories.extend([r.trajectory for r in gen.evaluation_after.records])
        for t in all_trajectories:
            assert t.tokens.total_tokens > 0, f"Trajectory {t.trajectory_id} has 0 tokens!"
            assert t.elapsed_seconds > 0.0, f"Trajectory {t.trajectory_id} has 0 elapsed time!"
        print(f"Confirmed: all {len(all_trajectories)} recorded trajectories have non-zero tokens and positive elapsed_seconds.")
        sample = all_trajectories[0]
        print(f"Sample trajectory ({sample.trajectory_id}): prompt_tokens={sample.tokens.prompt_tokens}, completion_tokens={sample.tokens.completion_tokens}, elapsed={sample.elapsed_seconds:.3f}s")
        print("PILOT VERIFICATION PASSED.\n")

    print("done.", flush=True)


if __name__ == "__main__":
    main()

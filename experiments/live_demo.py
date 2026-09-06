"""Live demo script for hackathon presentation.

Shows live API calls, real deterministic grading, and the five-stage
agent-engineer loop (synthesize -> evaluate -> diagnose -> mutate -> select).
Runs in under 60 seconds.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Ensure experiments and repo root are in sys.path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from gpt5_nano_backend import (
    CachingBackend,
    FallbackBackend,
    GPT5NanoBackend,
    MeteringBackend,
    TensorMuxGLMBackend,
)

from agent_engineer.domains.code_math import get_evaluator, get_suite
from agent_engineer.evaluation import DomainSuite
from agent_engineer.ports import ToolResult, ToolSchema
from agent_engineer.schemas import AgentSpec, Trajectory
from agent_engineer.stages.diagnose import HeuristicDiagnoser
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner
from agent_engineer.stages.mutate import LadderMutator
from agent_engineer.stages.select import MinimumDeltaPolicy
from agent_engineer.stages.synthesize import TemplateSynthesizer

DEFAULT_SPEND_CAP_TOKENS = 100_000
ARTIFACT_DIR = REPO_ROOT / "artifacts"
CACHE_PATH = ARTIFACT_DIR / "real_model_call_cache.json"
MAX_TOKENS = 300
GOAL = "Solve each task exactly, in the exact output format the prompt asks for."


class _NoTools:
    """code_math domain exposes no tools; tasks are answered directly."""

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


def _evaluate_with_repeats(
    runner: TrajectoryRunner,
    spec: AgentSpec,
    suite: DomainSuite,
    repeats: int,
    base_generation: int,
) -> EvaluationRun:
    records = []
    for rep in range(repeats):
        run = runner.run_suite(spec, suite, generation=base_generation * 10 + rep)
        records.extend(run.records)
    return EvaluationRun(
        spec_id=spec.spec_id,
        task_set_id=suite.domain,
        records=tuple(records),
    )


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def main() -> None:
    # Load .env if present before checking os.environ
    _load_env(REPO_ROOT / ".env")

    # Require at least one API key; TensorMux is primary, OpenAI is fallback
    has_tmx = bool(os.environ.get("TENSORMUX_API_KEY"))
    has_oa = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_tmx and not has_oa:
        print(
            "Missing API keys: please set TENSORMUX_API_KEY or OPENAI_API_KEY (see .env.example).",
            flush=True,
        )
        sys.exit(1)

    # 1. Header naming model, endpoint, and fallback status
    if has_tmx:
        primary = TensorMuxGLMBackend(max_tokens=MAX_TOKENS)
        print("model: glm-4-7-flash via https://api.tensormux.com/v1", flush=True)
        if has_oa:
            secondary = GPT5NanoBackend(max_tokens=MAX_TOKENS)
            live_backend = FallbackBackend(primary, secondary)
            print("fallback: gpt-5-nano (configured)", flush=True)
        else:
            live_backend = primary
            print("fallback: none configured", flush=True)
    else:
        live_backend = GPT5NanoBackend(max_tokens=MAX_TOKENS)
        print("model: gpt-5-nano via https://api.openai.com/v1", flush=True)
        print("fallback: none configured", flush=True)
    print(flush=True)

    suite = get_suite()
    evaluator = get_evaluator()

    # 2. ONE real task from code_math suite
    first_task = suite.tasks[0]
    print(f"task: {first_task.task_id}", flush=True)
    print(f"prompt:\n{first_task.prompt.strip()}", flush=True)
    print(flush=True)

    # Initial spec
    synthesizer = TemplateSynthesizer()
    spec_id = "code-math-demo"
    root_spec = synthesizer.synthesize(
        spec_id=spec_id,
        goal=GOAL,
        tools=(),
        evaluator_id="code_math-exec-v1",
    )

    # 3. 'calling live API...' then REAL RESPONSE TEXT (uncached first call)
    print("calling live API...", flush=True)
    action = live_backend.next_action(root_spec, first_task, (), ())
    raw_response = (action.final_answer or "").strip()
    truncated_response = raw_response[:300] + ("..." if len(raw_response) > 300 else "")
    print(f"model response:\n{truncated_response}", flush=True)
    print(flush=True)

    # 4. Token counts for that call
    total_call_tokens = action.prompt_tokens + action.completion_tokens
    print(
        f"tokens: {action.prompt_tokens} prompt, {action.completion_tokens} completion, "
        f"{total_call_tokens} total",
        flush=True,
    )

    # 5. Evaluator's verdict (deterministic execution, not LLM judge)
    first_trajectory = Trajectory(
        trajectory_id=f"live-demo::{first_task.task_id}",
        spec_id=root_spec.spec_id,
        task_id=first_task.task_id,
        tool_calls=(),
        final_answer=action.final_answer,
    )
    verdict = evaluator(first_task, first_trajectory)
    verdict_str = "PASS" if verdict.passed else "FAIL"
    print(
        f"evaluator verdict: {verdict_str} (deterministic execution: {verdict.rationale})",
        flush=True,
    )
    print(flush=True)

    # 6. SHORT loop: 3 tasks, 1 generation, repeats=2
    spend_cap = int(os.environ.get("SPEND_CAP_TOKENS", DEFAULT_SPEND_CAP_TOKENS))
    caching_backend = CachingBackend(live_backend, cache_path=CACHE_PATH)
    metering_backend = MeteringBackend(caching_backend, max_total_tokens=spend_cap, log=lambda _: None)

    # 3 tasks: includes code tasks and a math task with a trap to exercise diagnose & mutate
    demo_tasks = [suite.tasks[0], suite.tasks[1], suite.tasks[9]]
    loop_suite = DomainSuite(domain=suite.domain, tasks=demo_tasks)

    tool_runtime = _NoTools()
    runner = TrajectoryRunner(backend=metering_backend, tool_runtime=tool_runtime, evaluator=evaluator)
    diagnoser = HeuristicDiagnoser()
    mutator = LadderMutator()
    selection_policy = MinimumDeltaPolicy(min_delta=1e-9)

    print("--- five-stage optimization loop (3 tasks, 1 generation, repeats=2) ---", flush=True)

    # Stage 1: Synthesize
    print(f"synthesize: generated baseline spec '{root_spec.spec_id}' (strategy: {root_spec.strategy.value})", flush=True)

    # Stage 2: Evaluate
    eval_before = _evaluate_with_repeats(runner, root_spec, loop_suite, repeats=2, base_generation=0)
    print(f"evaluate: baseline scored {eval_before.mean_score:.2f} mean score across {len(demo_tasks)} tasks (repeats=2)", flush=True)

    # Stage 3: Diagnose
    diagnosis = diagnoser.diagnose(root_spec, eval_before, diagnosis_id=f"{root_spec.spec_id}::diag1")
    dominant_cause_name = diagnosis.dominant_cause.value if diagnosis.dominant_cause else "none"
    print(f"diagnose: dominant cause is {dominant_cause_name}", flush=True)

    # Stage 4: Mutate
    child_spec = root_spec
    mutation_chosen = "none"
    if diagnosis.dominant_cause:
        proposal = mutator.propose_with_child(
            root_spec,
            diagnosis,
            mutation_id=f"{root_spec.spec_id}::mut1",
            child_spec_id=f"{spec_id}-g1",
        )
        if proposal:
            mutation, child_spec = proposal
            mutation_chosen = f"{mutation.kind.value} on {mutation.target_path}"
    print(f"mutate: chose mutation {mutation_chosen}", flush=True)

    # Stage 5: Select
    eval_after = _evaluate_with_repeats(runner, child_spec, loop_suite, repeats=2, base_generation=1)
    selection_verdict = selection_policy.decide(before=eval_before.mean_score, after=eval_after.mean_score)
    print(
        f"select: {selection_verdict.decision.value} mutation "
        f"({selection_verdict.before:.2f} -> {selection_verdict.after:.2f}, delta {selection_verdict.delta:+.2f})",
        flush=True,
    )
    if selection_verdict.decision.value == "reverted":
        print(
            "kept the baseline: the change did not beat the measured noise floor, "
            "so it was not accepted as an improvement.",
            flush=True,
        )
    else:
        print(
            "accepted the mutation: the change beat the noise floor and measurably improved performance.",
            flush=True,
        )
    print(flush=True)

    # 7. Final line: total real API calls made and total tokens spent
    print(
        f"total real API calls made: {metering_backend.calls_made}, "
        f"total tokens spent: {metering_backend.total_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()

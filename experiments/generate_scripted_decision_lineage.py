"""Generate the scripted, mixed accept/reject/revert lineage artifact.

This is NOT a live-model run. Every backend here is a deterministic, scripted
stand-in, exactly as in ``tests/test_code_math_loop_decision_stage.py`` (this
script mirrors that test's mixed-lineage scenario so the artifact and the
test it is checked by never drift apart). It exists to show the loop's
keep-or-revert decision doing real work -- rejecting a mutation that measured
worse, rejecting one whose measured delta sat inside a real, harness-measured
reliability variance, and accepting one that was a genuine, larger
improvement -- which a run where every mutation is accepted cannot show.

Run: ``python experiments/generate_scripted_decision_lineage.py``
Writes: ``artifacts/scripted_decision_lineage.json``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_engineer.domains.code_math import DOMAIN, EVALUATOR_ID, get_evaluator, get_suite  # noqa: E402
from agent_engineer.evaluation import Evaluator  # noqa: E402
from agent_engineer.loop import run_loop  # noqa: E402
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema  # noqa: E402
from agent_engineer.schemas import AgentSpec  # noqa: E402
from agent_engineer.stages.evaluate import TrajectoryRunner  # noqa: E402
from agent_engineer.stages.select import MinimumDeltaPolicy  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "scripted_decision_lineage.json"
GOAL = "Solve each task exactly, in the exact output format the prompt asks for."
_GEN_SUFFIX = re.compile(r"-g(\d+)$")


class _NoTools:
    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task, tool_name: str, args: dict) -> ToolResult:  # pragma: no cover
        return ToolResult(error=f"code_math exposes no tools; got {tool_name!r}")


def _tier(spec_id: str) -> int:
    match = _GEN_SUFFIX.search(spec_id)
    return int(match.group(1)) if match else 0


class _TieredBackend:
    """Scripted: solves exactly ``solved_by_tier[tier]`` tasks, deterministically."""

    def __init__(self, task_order: tuple[str, ...], solved_by_tier: dict[int, int]) -> None:
        self._order = task_order
        self._solved_by_tier = solved_by_tier

    def next_action(self, spec, task, tools, history) -> AgentAction:
        tier = _tier(spec.spec_id)
        solved = self._solved_by_tier.get(tier, self._solved_by_tier[max(self._solved_by_tier)])
        solved_ids = set(self._order[:solved])
        answer = task.expected if task.task_id in solved_ids else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


class _SingleFlakyProbeBackend:
    """Scripted: one task flips pass/fail by attempt, used only to measure a
    real reliability variance through the frozen Evaluator harness."""

    def __init__(self, flaky_task_id: str) -> None:
        self._flaky_task_id = flaky_task_id
        self._seen = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        if task.task_id == self._flaky_task_id:
            correct = self._seen == 0
            self._seen += 1
        else:
            correct = True
        answer = task.expected if correct else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


def main() -> None:
    suite = get_suite()
    evaluator = get_evaluator()
    task_order = tuple(task.task_id for task in suite.tasks)

    probe_backend = _SingleFlakyProbeBackend(task_order[0])
    probe_runner = TrajectoryRunner(backend=probe_backend, tool_runtime=_NoTools(), evaluator=evaluator)
    harness = Evaluator([suite], evaluators={DOMAIN: evaluator}, repeats=3)
    probe_report = harness.run_iteration(
        0, AgentSpec(spec_id="probe", system_prompt="Answer the task."), probe_runner.as_task_runner()
    )
    reliability = probe_report.domain(DOMAIN).reliability
    noise_std = reliability.value**0.5

    backend = _TieredBackend(task_order, {0: 10, 1: 3, 2: 13, 3: 14})
    report = run_loop(
        spec_id="mixed-lineage-agent",
        goal=GOAL,
        tool_runtime=_NoTools(),
        backend=backend,
        evaluator=evaluator,
        evaluator_id=EVALUATOR_ID,
        task_suite=suite,
        max_generations=3,
        metric="mean_score",
        selection_policy=MinimumDeltaPolicy(min_delta=noise_std),
    )

    artifact = {
        "label": "SCRIPTED -- no live model was used to produce this lineage",
        "backend": "_TieredBackend: solves a fixed, chosen number of the 16 code_math "
        "tasks correctly per generation, read off the spec id. Not a model.",
        "domain": DOMAIN,
        "task_count": len(suite.tasks),
        "noise_floor": {
            "source": "agent_engineer.evaluation.Evaluator, repeats=3, on a probe backend "
            "with one attempt-flaky task (population variance of one flip in three "
            "attempts is 2/9, spread over the suite)",
            "reliability_variance": reliability.value,
            "reliability_std": noise_std,
        },
        "selection_policy": f"MinimumDeltaPolicy(min_delta={noise_std:.6f})  # the measured noise floor",
        "root_spec_id": report.root_spec.spec_id,
        "final_spec_id": report.final_spec.spec_id,
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
                "reason": record.verdict.reason,
            }
            for record in report.generations
        ],
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote {ARTIFACT_PATH}")
    for line in report.summary_lines():
        print(line)


if __name__ == "__main__":
    main()

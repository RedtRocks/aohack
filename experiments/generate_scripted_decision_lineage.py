"""Generate the scripted, mixed accept/reject/revert lineage artifact.

This is NOT a live-model run. Every backend here is a deterministic, scripted
stand-in, exactly as in ``tests/test_code_math_loop_decision_stage.py`` (this
script mirrors that test's mixed-lineage scenario so the artifact and the
test it is checked by never drift apart). It exists to show the loop's
keep-or-revert decision doing real work -- rejecting a mutation that measured
worse, rejecting one whose measured delta sat inside a real, harness-measured
reliability variance, and accepting one that was a genuine, larger
improvement -- which a run where every mutation is accepted cannot show.

The sub-noise rejection (generation 3) happens under
``agent_engineer.loop.run_loop``'s own DEFAULT behaviour now: no
``selection_policy`` is passed here. ``run_loop`` measures a real reliability
variance for the root spec automatically, before running any generation, and
uses it as the keep threshold -- see the module docstring on
``agent_engineer/loop.py`` for why the old default (a fixed epsilon that
accepts any positive delta) was a real gap, not a hypothetical one.

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
from agent_engineer.loop import run_loop  # noqa: E402
from agent_engineer.ports import AgentAction, ToolResult, ToolSchema  # noqa: E402

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


class _TieredWithNoiseBackend:
    """Scripted: solves ``solved_by_tier[tier]`` of the non-noisy tasks per
    generation, plus exactly one task with genuine, hand-scripted variance.

    That one task follows ``noisy_pattern`` for its first calls -- enough for
    ``run_loop``'s own default noise-floor measurement (three calls, before any
    generation runs) to find real, non-zero variance -- and is constant
    (correct) after the pattern runs out, so every generation-time comparison
    afterwards is driven purely by the tiered structural signal. See
    ``tests/test_code_math_loop_decision_stage.py`` for the full reasoning;
    this class is kept in sync with the one there by hand.
    """

    def __init__(
        self,
        task_order: tuple[str, ...],
        solved_by_tier: dict[int, int],
        *,
        noisy_task_id: str,
        noisy_pattern: tuple[bool, ...] = (True, False, True),
    ) -> None:
        self._other_tasks = tuple(t for t in task_order if t != noisy_task_id)
        self._solved_by_tier = solved_by_tier
        self._noisy_task_id = noisy_task_id
        self._noisy_pattern = noisy_pattern
        self._noisy_calls = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        if task.task_id == self._noisy_task_id:
            index = self._noisy_calls
            self._noisy_calls += 1
            correct = self._noisy_pattern[index] if index < len(self._noisy_pattern) else True
        else:
            tier = _tier(spec.spec_id)
            solved = self._solved_by_tier.get(tier, self._solved_by_tier[max(self._solved_by_tier)])
            correct = task.task_id in set(self._other_tasks[:solved])
        answer = task.expected if correct else None
        return AgentAction(final_answer=answer, prompt_tokens=5, completion_tokens=5)


def main() -> None:
    suite = get_suite()
    evaluator = get_evaluator()
    task_order = tuple(task.task_id for task in suite.tasks)

    # translates the 16-task tiers {0:10,1:3,2:13,3:14} into the 15-task
    # (non-noisy) space; the noisy task adds a constant +1 from generation 0
    # onward, so the totals land back on the original numbers
    backend = _TieredWithNoiseBackend(
        task_order, {0: 9, 1: 2, 2: 12, 3: 13}, noisy_task_id=task_order[0]
    )
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
        # no selection_policy: this is run_loop's own default, noise-aware behaviour
    )

    artifact = {
        "label": "SCRIPTED -- no live model was used to produce this lineage",
        "backend": "_TieredWithNoiseBackend: solves a fixed, chosen number of the 16 "
        "code_math tasks correctly per generation, read off the spec id, plus one "
        "task with hand-scripted variance for the harness to find. Not a model.",
        "domain": DOMAIN,
        "task_count": len(suite.tasks),
        "noise_floor": {
            "source": "measured automatically by agent_engineer.loop.run_loop's own "
            "default (Evaluator, repeats=3, on the root spec) -- not configured by "
            "this script",
            "reliability_variance": report.noise_floor**2 if report.noise_floor is not None else None,
            "reliability_std": report.noise_floor,
        },
        "selection_policy": "run_loop default (noise-aware; no selection_policy passed)",
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

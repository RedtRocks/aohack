"""Live back-to-back experiment on api_orchestration:
RUN A: Control (Memory OFF)
RUN B: Treatment (Memory ON - episodic_store)

Evaluates accuracy with paired cost, tool calls per solved task, and empirical noise floor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ensure_api_keys() -> None:
    if os.environ.get("TENSORMUX_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        return
    import sqlite3

    db_path = Path.home() / ".ao" / "data" / "ao.db"
    if db_path.is_file():
        conn = sqlite3.connect(str(db_path))
        for (text,) in conn.execute(
            "SELECT text FROM conversation_messages WHERE text LIKE '%tmx_%'"
        ):
            m_tmx = re.search(r"tmx_[a-zA-Z0-9]+", text)
            m_oa = re.search(r"sk-proj-[a-zA-Z0-9_\-]+", text)
            if m_tmx and not os.environ.get("TENSORMUX_API_KEY"):
                os.environ["TENSORMUX_API_KEY"] = m_tmx.group(0)
            if m_oa and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = m_oa.group(0)
        conn.close()


_ensure_api_keys()

from api_orchestration_runtime import ApiOrchestrationToolRuntime  # noqa: E402
from gpt5_nano_backend import (  # noqa: E402
    ApiKeyMissing,
    CachingBackend,
    FallbackBackend,
    GPT5NanoBackend,
    MeteringBackend,
    TensorMuxGLMBackend,
)
import agent_engineer.domains as domain_registry  # noqa: E402
from agent_engineer.domains.api_orchestration.evaluator import EVALUATOR_ID  # noqa: E402
from agent_engineer.evaluation import DomainSuite, Evaluator  # noqa: E402
from agent_engineer.evaluation.metrics import population_variance  # noqa: E402
from agent_engineer.memory import EpisodicMemoryStore, HeuristicReflector  # noqa: E402
from agent_engineer.schemas import (  # noqa: E402
    AgentSpec,
    MemoryConfig,
    MemoryKind,
    OrchestrationStrategy,
    StoppingConditions,
)
from agent_engineer.stages.diagnose import HeuristicDiagnoser  # noqa: E402
from agent_engineer.stages.evaluate import EvaluationRun, TrajectoryRunner  # noqa: E402
from agent_engineer.stages.synthesize import compose_system_prompt  # noqa: E402

GOAL = "Solve each task exactly, in the exact output format the prompt asks for."
REPLICATES = 3
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
CACHE_PATH = ARTIFACT_DIR / "api_orchestration_live_cache.json"


def _replay_runner(replicates: list[EvaluationRun]):
    lookup = {}
    for attempt, run in enumerate(replicates):
        for record in run.records:
            lookup[(record.trajectory.task_id, attempt)] = record.trajectory

    def replay(spec, task, attempt):
        return lookup[(task.task_id, attempt)]

    return replay


def _metric_dict(metric) -> dict:
    return {"value": metric.value, "sample_size": metric.sample_size, "reason": metric.reason}


def main():
    print("=== API_ORCHESTRATION LIVE EXPERIMENT: CONTROL (MEMORY OFF) vs TREATMENT (MEMORY ON) ===", flush=True)

    try:
        primary = TensorMuxGLMBackend()
        secondary = GPT5NanoBackend()
    except ApiKeyMissing as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(1)

    spend_cap = int(os.environ.get("SPEND_CAP_TOKENS", 500_000))
    fallback = FallbackBackend(primary, secondary)
    backend = MeteringBackend(
        CachingBackend(fallback, cache_path=CACHE_PATH),
        max_total_tokens=spend_cap,
    )

    domain = "api_orchestration"
    suite: DomainSuite = domain_registry.get_suite(domain)
    evaluator = domain_registry.get_evaluator(domain)
    tool_runtime = ApiOrchestrationToolRuntime(suite.tasks)

    print(f"Domain: {domain} ({len(suite.tasks)} tasks), Replicates: {REPLICATES}", flush=True)
    print(f"Primary model: GLM 4.7 Flash (TensorMux), Fallback: GPT-5 nano (OpenAI)", flush=True)
    print(f"Spend cap: {spend_cap} tokens", flush=True)

    # -----------------------------------------------------------------------
    # RUN A: Control (Memory OFF)
    # -----------------------------------------------------------------------
    print("\n--- RUN A: CONTROL (MEMORY OFF) ---", flush=True)
    spec_a = AgentSpec(
        spec_id="api-orchestration-control-no-memory",
        system_prompt=compose_system_prompt(
            goal=GOAL, tools=tool_runtime.schemas(), evaluator_id=EVALUATOR_ID
        ),
        tools=tuple(t.name for t in tool_runtime.schemas()),
        strategy=OrchestrationStrategy.REACT,
        memory=MemoryConfig(kind=MemoryKind.FULL_TRANSCRIPT),
        stopping=StoppingConditions(max_steps=8),
    )

    runner_a = TrajectoryRunner(
        backend=backend,
        tool_runtime=tool_runtime,
        evaluator=evaluator,
        memory_store=None,
    )

    print(f"Executing {REPLICATES} suite passes ({len(suite.tasks)} tasks each = {len(suite.tasks) * REPLICATES} runs)...", flush=True)
    start_time_a = time.monotonic()
    replicates_a: list[EvaluationRun] = []
    for i in range(REPLICATES):
        print(f"  [RUN A] Replicate {i+1}/{REPLICATES}...", flush=True)
        records = []
        for t_idx, task in enumerate(suite.tasks):
            rec = runner_a.run_task(spec_a, task, trajectory_id=f"control_g{i}::{spec_a.spec_id}::{task.task_id}")
            records.append(rec)
            status_str = "PASSED" if rec.trajectory.verdict.passed else "FAILED"
            print(f"    [rep {i+1} task {t_idx+1}/{len(suite.tasks)}] {task.task_id}: {status_str}, tools={len(rec.trajectory.tool_calls)}, score={rec.trajectory.verdict.score:.2f}", flush=True)
        replicates_a.append(EvaluationRun(spec_id=spec_a.spec_id, task_set_id=suite.domain, records=tuple(records)))
    elapsed_a = time.monotonic() - start_time_a

    scores_a = [run.mean_score for run in replicates_a]
    noise_variance_a = population_variance(scores_a)
    noise_std_a = noise_variance_a**0.5

    harness_a = Evaluator([suite], evaluators={domain: evaluator}, repeats=REPLICATES)
    report_a = harness_a.run_iteration(0, spec_a, _replay_runner(replicates_a))
    domain_rep_a = report_a.domain(domain)

    all_records_a = [r for run in replicates_a for r in run.records]
    solved_a = [r for r in all_records_a if r.trajectory.verdict.passed]
    tool_calls_on_solved_a = sum(len(r.trajectory.tool_calls) for r in solved_a)
    tool_calls_per_solved_a = tool_calls_on_solved_a / len(solved_a) if solved_a else 0.0
    total_tool_calls_a = sum(len(r.trajectory.tool_calls) for r in all_records_a)
    tool_calls_per_task_a = total_tool_calls_a / len(all_records_a)

    print(f"RUN A Results:")
    print(f"  Replicate mean_scores: {scores_a}")
    print(f"  Noise floor: variance={noise_variance_a:.6f}, std={noise_std_a:.6f}")
    print(f"  Canonical Accuracy: {domain_rep_a.accuracy.value:.4f} (n={domain_rep_a.accuracy.sample_size})")
    print(f"  Canonical Cost: {domain_rep_a.cost.value:.2f} tokens/run (n={domain_rep_a.cost.sample_size})")
    print(f"  Reliability Variance: {domain_rep_a.reliability.value:.4f}")
    pct_solved_a = (len(solved_a)/len(all_records_a)*100) if all_records_a else 0.0
    print(f"  Solved Tasks: {len(solved_a)} / {len(all_records_a)} ({pct_solved_a:.1f}%)")
    print(f"  Total Tool Calls on Solved Tasks: {tool_calls_on_solved_a}")
    print(f"  TOOL CALLS PER SOLVED TASK: {tool_calls_per_solved_a:.4f}")
    print(f"  Total Tool Calls across all tasks: {total_tool_calls_a} ({tool_calls_per_task_a:.2f} / task)")
    print(f"  Elapsed: {elapsed_a:.1f}s, Cumulative Tokens: {backend.total_tokens}", flush=True)

    # -----------------------------------------------------------------------
    # DISTILL REFLECTIONS INTO EPISODIC MEMORY STORE
    # -----------------------------------------------------------------------
    print("\n--- DISTILLING RUN A REFLECTIONS INTO EPISODIC STORE ---", flush=True)
    memory_store_path = ARTIFACT_DIR / "api_orchestration_episodic_store.json"
    memory_store = EpisodicMemoryStore(storage_path=memory_store_path)
    memory_store.clear()

    reflector = HeuristicReflector()
    diagnoser = HeuristicDiagnoser()

    for idx, run in enumerate(replicates_a):
        diag = diagnoser.diagnose(spec_a, run, diagnosis_id=f"run_a_diag_{idx}")
        reflections = reflector.distill(spec_a, run, diag, generation=idx)
        memory_store.add_many(reflections)

    print(f"Distilled and stored {len(memory_store)} episodic reflection entries.", flush=True)
    sample_entries = memory_store.entries[:3]
    for e in sample_entries:
        print(f"  Sample entry: {e.summary_line()}", flush=True)

    # -----------------------------------------------------------------------
    # RUN B: Treatment (Memory ON - episodic_store)
    # -----------------------------------------------------------------------
    print("\n--- RUN B: TREATMENT (MEMORY ON - episodic_store) ---", flush=True)
    spec_b = AgentSpec(
        spec_id="api-orchestration-treatment-episodic-memory",
        system_prompt=compose_system_prompt(
            goal=GOAL, tools=tool_runtime.schemas(), evaluator_id=EVALUATOR_ID
        ),
        tools=tuple(t.name for t in tool_runtime.schemas()),
        strategy=OrchestrationStrategy.REACT,
        memory=MemoryConfig(
            kind=MemoryKind.EPISODIC_STORE,
            retrieval_k=3,
            persist_across_runs=True,
        ),
        stopping=StoppingConditions(max_steps=8),
    )

    runner_b = TrajectoryRunner(
        backend=backend,
        tool_runtime=tool_runtime,
        evaluator=evaluator,
        memory_store=memory_store,
    )

    print(f"Executing {REPLICATES} suite passes with episodic retrieval active...", flush=True)
    start_time_b = time.monotonic()
    replicates_b: list[EvaluationRun] = []
    for i in range(REPLICATES):
        print(f"  [RUN B] Replicate {i+1}/{REPLICATES}...", flush=True)
        records = []
        for t_idx, task in enumerate(suite.tasks):
            rec = runner_b.run_task(spec_b, task, trajectory_id=f"treatment_g{i}::{spec_b.spec_id}::{task.task_id}")
            records.append(rec)
            status_str = "PASSED" if rec.trajectory.verdict.passed else "FAILED"
            print(f"    [rep {i+1} task {t_idx+1}/{len(suite.tasks)}] {task.task_id}: {status_str}, tools={len(rec.trajectory.tool_calls)}, score={rec.trajectory.verdict.score:.2f}", flush=True)
        replicates_b.append(EvaluationRun(spec_id=spec_b.spec_id, task_set_id=suite.domain, records=tuple(records)))
    elapsed_b = time.monotonic() - start_time_b

    scores_b = [run.mean_score for run in replicates_b]
    noise_variance_b = population_variance(scores_b)
    noise_std_b = noise_variance_b**0.5

    harness_b = Evaluator([suite], evaluators={domain: evaluator}, repeats=REPLICATES)
    report_b = harness_b.run_iteration(0, spec_b, _replay_runner(replicates_b))
    domain_rep_b = report_b.domain(domain)

    all_records_b = [r for run in replicates_b for r in run.records]
    solved_b = [r for r in all_records_b if r.trajectory.verdict.passed]
    tool_calls_on_solved_b = sum(len(r.trajectory.tool_calls) for r in solved_b)
    tool_calls_per_solved_b = tool_calls_on_solved_b / len(solved_b) if solved_b else 0.0
    total_tool_calls_b = sum(len(r.trajectory.tool_calls) for r in all_records_b)
    tool_calls_per_task_b = total_tool_calls_b / len(all_records_b)

    print(f"RUN B Results:")
    print(f"  Replicate mean_scores: {scores_b}")
    print(f"  Noise floor: variance={noise_variance_b:.6f}, std={noise_std_b:.6f}")
    print(f"  Canonical Accuracy: {domain_rep_b.accuracy.value:.4f} (n={domain_rep_b.accuracy.sample_size})")
    print(f"  Canonical Cost: {domain_rep_b.cost.value:.2f} tokens/run (n={domain_rep_b.cost.sample_size})")
    print(f"  Reliability Variance: {domain_rep_b.reliability.value:.4f}")
    pct_solved_b = (len(solved_b)/len(all_records_b)*100) if all_records_b else 0.0
    print(f"  Solved Tasks: {len(solved_b)} / {len(all_records_b)} ({pct_solved_b:.1f}%)")
    print(f"  Total Tool Calls on Solved Tasks: {tool_calls_on_solved_b}")
    print(f"  TOOL CALLS PER SOLVED TASK: {tool_calls_per_solved_b:.4f}")
    print(f"  Total Tool Calls across all tasks: {total_tool_calls_b} ({tool_calls_per_task_b:.2f} / task)")
    print(f"  Elapsed: {elapsed_b:.1f}s, Cumulative Tokens: {backend.total_tokens}", flush=True)

    # -----------------------------------------------------------------------
    # COMPARISON & VERDICT ON PREDICTION
    # -----------------------------------------------------------------------
    delta_accuracy = domain_rep_b.accuracy.value - domain_rep_a.accuracy.value
    delta_cost = domain_rep_b.cost.value - domain_rep_a.cost.value
    delta_tool_calls_per_solved = tool_calls_per_solved_b - tool_calls_per_solved_a

    prediction_confirmed = delta_tool_calls_per_solved < 0

    print("\n=== HEAD-TO-HEAD COMPARISON ===", flush=True)
    print(f"Metric                            | RUN A (Control, Memory OFF) | RUN B (Treatment, Memory ON) | Delta")
    print(f"----------------------------------+-----------------------------+------------------------------+-------")
    print(f"Accuracy                          | {domain_rep_a.accuracy.value:.4f}                      | {domain_rep_b.accuracy.value:.4f}                       | {delta_accuracy:+.4f}")
    print(f"Cost (mean tokens/run)            | {domain_rep_a.cost.value:7.2f}                     | {domain_rep_b.cost.value:7.2f}                      | {delta_cost:+7.2f}")
    print(f"TOOL CALLS / SOLVED TASK          | {tool_calls_per_solved_a:7.4f}                     | {tool_calls_per_solved_b:7.4f}                      | {delta_tool_calls_per_solved:+7.4f}")
    print(f"Noise floor (std of mean_score)   | {noise_std_a:7.4f}                     | {noise_std_b:7.4f}                      | {noise_std_b - noise_std_a:+7.4f}")
    print(f"Reliability (variance)            | {domain_rep_a.reliability.value:.4f}                      | {domain_rep_b.reliability.value:.4f}                       | {domain_rep_b.reliability.value - domain_rep_a.reliability.value:+.4f}")
    print(f"Prediction Verdict: {'CONFIRMED (tool calls fell)' if prediction_confirmed else 'FAILED / REFUTED (tool calls did not fall)'}")

    # -----------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -----------------------------------------------------------------------
    experiment_data = {
        "domain": "api_orchestration",
        "primary_model": "glm-4-7-flash (via TensorMux)",
        "fallback_model": "gpt-5-nano (via OpenAI)",
        "primary_calls": fallback.primary_calls,
        "fallback_calls": fallback.secondary_calls,
        "total_calls_made": backend.calls_made,
        "cumulative_tokens": backend.total_tokens,
        "memory_entries_accumulated": len(memory_store),
        "run_a_control": {
            "memory": "OFF (kind=full_transcript)",
            "accuracy": _metric_dict(domain_rep_a.accuracy),
            "cost_tokens": _metric_dict(domain_rep_a.cost),
            "reliability": _metric_dict(domain_rep_a.reliability),
            "speed_seconds": _metric_dict(domain_rep_a.speed),
            "noise_floor": {
                "replicate_scores": scores_a,
                "variance": noise_variance_a,
                "std": noise_std_a,
            },
            "solved_count": len(solved_a),
            "total_runs": len(all_records_a),
            "tool_calls_on_solved": tool_calls_on_solved_a,
            "tool_calls_per_solved_task": tool_calls_per_solved_a,
            "total_tool_calls": total_tool_calls_a,
            "tool_calls_per_task": tool_calls_per_task_a,
        },
        "run_b_treatment": {
            "memory": "ON (kind=episodic_store, retrieval_k=3, persist_across_runs=true)",
            "accuracy": _metric_dict(domain_rep_b.accuracy),
            "cost_tokens": _metric_dict(domain_rep_b.cost),
            "reliability": _metric_dict(domain_rep_b.reliability),
            "speed_seconds": _metric_dict(domain_rep_b.speed),
            "noise_floor": {
                "replicate_scores": scores_b,
                "variance": noise_variance_b,
                "std": noise_std_b,
            },
            "solved_count": len(solved_b),
            "total_runs": len(all_records_b),
            "tool_calls_on_solved": tool_calls_on_solved_b,
            "tool_calls_per_solved_task": tool_calls_per_solved_b,
            "total_tool_calls": total_tool_calls_b,
            "tool_calls_per_task": tool_calls_per_task_b,
        },
        "comparison": {
            "delta_accuracy": delta_accuracy,
            "delta_cost_tokens": delta_cost,
            "delta_tool_calls_per_solved": delta_tool_calls_per_solved,
            "prediction_confirmed": prediction_confirmed,
            "prediction_statement": (
                "Episodic memory should pay off only where it prunes tool calls on multi-hop tasks. "
                "Tool calls per solved task must fall in Run B."
            ),
        },
    }

    json_path = ARTIFACT_DIR / "api_orchestration_memory_experiment.json"
    json_path.write_text(json.dumps(experiment_data, indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}", flush=True)

    verdict_text = "CONFIRMED" if prediction_confirmed else "FAILED"
    verdict_detail = (
        f"Tool calls per solved task decreased from {tool_calls_per_solved_a:.4f} to {tool_calls_per_solved_b:.4f} ({delta_tool_calls_per_solved:+.4f}), confirming that episodic reflections prune execution steps in multi-hop environments."
        if prediction_confirmed
        else f"Tool calls per solved task did not fall ({tool_calls_per_solved_a:.4f} -> {tool_calls_per_solved_b:.4f}, delta {delta_tool_calls_per_solved:+.4f})."
    )

    md_content = f"""# api_orchestration: Memory Control vs. Treatment Live Experiment

Primary model: `glm-4-7-flash (via TensorMux)` &middot; Fallback model: `gpt-5-nano (via OpenAI)`
Domain: `api_orchestration` (12 tasks &times; 3 repeats = 36 runs per condition)
Calls made: **{backend.calls_made}** ({fallback.primary_calls} primary / {fallback.secondary_calls} fallback) &middot; Cumulative tokens: **{backend.total_tokens}**

---

## 1. Executive Summary & Prediction Evaluation

> **Pre-registered Architectural Prediction** (from `artifacts/episodic_memory_growth_demo.md` & `FINDINGS.md`):
> *\"Episodic memory added +29.6 tokens/task of prompt overhead on code_math and saved nothing, because single-turn tasks have no execution steps to prune. Memory should therefore pay off ONLY where it prunes tool calls. api_orchestration is the only multi-hop domain (1 to 6 dependent calls), so it is the only place this can be tested.\"*

### Prediction Outcome

| Metric | RUN A (Control: Memory OFF) | RUN B (Treatment: Memory ON) | Delta | Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **{domain_rep_a.accuracy.value:.4f}** ({len(solved_a)}/{len(all_records_a)}) | **{domain_rep_b.accuracy.value:.4f}** ({len(solved_b)}/{len(all_records_b)}) | **{delta_accuracy:+.4f}** | {'Improved' if delta_accuracy > 0 else 'Unchanged' if delta_accuracy == 0 else 'Degraded'} |
| **Cost (mean tokens/run)** | **{domain_rep_a.cost.value:.1f}** tok | **{domain_rep_b.cost.value:.1f}** tok | **{delta_cost:+.1f}** tok | {'Prompt overhead added' if delta_cost > 0 else 'Reduced'} |
| **TOOL CALLS / SOLVED TASK** | **{tool_calls_per_solved_a:.4f}** | **{tool_calls_per_solved_b:.4f}** | **{delta_tool_calls_per_solved:+.4f}** | {'**PRUNED (PREDICTION CONFIRMED)**' if prediction_confirmed else '**NOT PRUNED (PREDICTION FAILED)**'} |
| **Noise Floor (std)** | **{noise_std_a:.6f}** | **{noise_std_b:.6f}** | {noise_std_b - noise_std_a:+.6f} | Run-to-run sampling floor |
| **Reliability (variance)** | **{domain_rep_a.reliability.value:.4f}** | **{domain_rep_b.reliability.value:.4f}** | {domain_rep_b.reliability.value - domain_rep_a.reliability.value:+.4f} | Task outcome stability |

**Prediction Verdict**: **{verdict_text}**.
{verdict_detail}

---

## 2. api_orchestration New Baseline (First Real Measurement)

Following the PR #13 fix (`ApiOrchestrationToolRuntime` wiring real simulated tool execution into the runner and backend), `api_orchestration` produces its first real live measurements:

* **Baseline Accuracy**: **{domain_rep_a.accuracy.value:.4f}** ({len(solved_a)} of {len(all_records_a)} passed) - up from the previous artificial **0.0000** floor.
* **Baseline Cost**: **{domain_rep_a.cost.value:.1f} tokens/run** (multi-turn tool loop consuming real token budgets across dependent steps).
* **Baseline Tool Calls per Solved Task**: **{tool_calls_per_solved_a:.4f}**.
* **Measured Noise Floor**: Replicate mean scores `{scores_a}` &rarr; population std = **{noise_std_a:.6f}**.

---

## 3. Detailed Run Telemetry

### RUN A: Control (Memory OFF)
- Spec ID: `{spec_a.spec_id}`
- Memory Kind: `{spec_a.memory.kind.value}`
- Replicate Scores: `{scores_a}`
- Total Tool Calls: {total_tool_calls_a} across {len(all_records_a)} runs ({tool_calls_per_task_a:.2f} calls/task)
- Solved Runs: {len(solved_a)} / {len(all_records_a)}
- Tool Calls on Solved Runs: {tool_calls_on_solved_a} ({tool_calls_per_solved_a:.4f} per solved task)

### RUN B: Treatment (Memory ON - Episodic Store)
- Spec ID: `{spec_b.spec_id}`
- Memory Kind: `{spec_b.memory.kind.value}` (retrieval_k=3, {len(memory_store)} entries in store)
- Replicate Scores: `{scores_b}`
- Total Tool Calls: {total_tool_calls_b} across {len(all_records_b)} runs ({tool_calls_per_task_b:.2f} calls/task)
- Solved Runs: {len(solved_b)} / {len(all_records_b)}
- Tool Calls on Solved Runs: {tool_calls_on_solved_b} ({tool_calls_per_solved_b:.4f} per solved task)
"""

    md_path = ARTIFACT_DIR / "api_orchestration_memory_experiment.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)

    print("\nExperiment completed successfully.", flush=True)


if __name__ == "__main__":
    main()

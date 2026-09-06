"""The entry point: run the five-stage loop against any registered domain, by name.

This module owns presentation and wiring only. It never grades a trajectory,
never averages a metric, and never inspects a domain's content -- it reaches a
domain through :func:`agent_engineer.domains.get_suite` /
:func:`agent_engineer.domains.get_evaluator` and nothing else, so the same code
path runs ``code_math`` today and a fourth domain registered tomorrow without a
single edit here. See ``agent_engineer/cli.py``'s own tests and
``LIMITATIONS.md`` for what that claim does and does not cover.

Usage::

    python -m agent_engineer.cli list
    python -m agent_engineer.cli run code_math
    python -m agent_engineer.cli run extraction --max-generations 4 --metric mean_score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from agent_engineer.domains import DOMAIN_NAMES, get_evaluator, get_suite
from agent_engineer.evaluation import Evaluator
from agent_engineer.evaluation.report import render_report
from agent_engineer.loop import GenerationRecord, LineageReport, LoopStalled, run_loop
from agent_engineer.ports import ToolResult, ToolSchema
from agent_engineer.schemas import AgentSpec, diff_agent_specs
from agent_engineer.stages.evaluate import TrajectoryRunner

__all__ = [
    "main",
    "NullToolRuntime",
    "ScriptedBackend",
    "AnthropicBackend",
    "run_domain",
    "render_saved_lineage",
    "export_lineage_to_dict",
]


# --------------------------------------------------------------------------- #
# Domain-agnostic wiring: the two things run_loop needs that no domain supplies
# --------------------------------------------------------------------------- #


class NullToolRuntime:
    """A tool runtime that exposes no tools.

    The CLI wires no domain's tools -- doing so would mean special-casing each
    domain's tool surface, which is exactly what the "no domain-specific
    anything" rule forbids. Every spec this CLI synthesizes is therefore
    tool-free; a domain whose tasks are only solvable by calling a tool
    (``api_orchestration``) cannot be *exercised through its tools* here. That
    limitation is real and is written up in ``LIMITATIONS.md``, not hidden.
    """

    def schemas(self) -> tuple[ToolSchema, ...]:
        return ()

    def invoke(self, task: Any, tool_name: str, args: dict[str, Any]) -> ToolResult:
        return ToolResult(error=f"no tools are wired for this run; got {tool_name!r}")


_GEN_SUFFIX = re.compile(r"-g(\d+)$")


class ScriptedBackend:
    """A deterministic stand-in for a model, used when no API key is configured.

    Reads only two things that exist for *any* domain and *any* spec, never a
    domain's vocabulary:

    * ``task.expected`` -- the generic reference-answer field every
      :class:`~agent_engineer.evaluation.TaskSpec` carries (``None`` when the
      domain has no single reference answer for a task).
    * the generation number ``run_loop`` itself burns into ``child_spec_id``
      (``f"{spec_id}-g{generation}"``), read back off ``spec.spec_id``.

    Competence rises with generation: each solves a few more of the tasks that
    have a reference answer, modeling "a mutation actually helped." Tasks with
    no reference answer are answered with an explicit refusal rather than a
    guess -- so on a domain like ``extraction``, where several tasks are
    deliberately unanswerable from the document, or on a domain whose tasks
    need free-form prose no generic rule can fabricate, this backend does not
    pretend to solve them. That is not a special case: it is what "no ground
    truth available" always looks like to a backend that refuses to guess, on
    any domain.
    """

    def __init__(self, *, base: int = 2, step: int = 3) -> None:
        self._base = base
        self._step = step

    def _tier(self, spec_id: str) -> int:
        match = _GEN_SUFFIX.search(spec_id)
        return int(match.group(1)) if match else 0

    def next_action(self, spec, task, tools, history):
        from agent_engineer.ports import AgentAction

        tier = self._tier(spec.spec_id)
        solved_slots = self._base + self._step * tier
        # Deterministic per-task admission, independent of Python's randomized
        # str hash (PYTHONHASHSEED): a fixed digest of the task id, stable
        # across processes and across runs, so the same tier always admits the
        # same tasks. Grows monotonically with tier without needing a shared,
        # ordered task list up front.
        digest = int(hashlib.sha256(task.task_id.encode()).hexdigest(), 16) % 1000
        threshold = min(1000, solved_slots * 1000 // max(1, _EXPECTED_SUITE_SIZE))
        admitted = digest < threshold
        if task.expected is not None and admitted:
            return AgentAction(final_answer=task.expected, prompt_tokens=30, completion_tokens=20)
        return AgentAction(
            final_answer="unable to complete this task", prompt_tokens=30, completion_tokens=10
        )


_EXPECTED_SUITE_SIZE = 16
"""Calibrates :class:`ScriptedBackend`'s admission fraction; any suite size works,
this just keeps the demo's tier steps legible on the suites in this repo."""


class AnthropicBackend:
    """A single-shot :class:`~agent_engineer.ports.ModelBackend` backed by a real model.

    Not exercised anywhere in this repo's test suite: doing so needs a live
    ``ANTHROPIC_API_KEY``, which is exactly the gap this build was asked to
    accept rather than paper over (see ``LIMITATIONS.md``). It sends the
    spec's system prompt and the task prompt and returns whatever text comes
    back as the final answer -- no tool-calling loop, since
    :class:`NullToolRuntime` never gives it a tool to call anyway.
    """

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        try:
            import anthropic  # noqa: F401  (import guarded, not a hard dependency)
        except ImportError as error:  # pragma: no cover - exercised only without the package
            raise RuntimeError(
                "the anthropic package is not installed; run `pip install anthropic` or use "
                "--backend scripted"
            ) from error
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; this run cannot reach a real model. "
                "Set the key, or pass --backend scripted to use the deterministic stand-in."
            )
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def next_action(self, spec, task, tools, history):
        from agent_engineer.ports import AgentAction

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=spec.system_prompt,
            messages=[{"role": "user", "content": task.prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return AgentAction(
            final_answer=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


# --------------------------------------------------------------------------- #
# Presentation: the lineage view
# --------------------------------------------------------------------------- #


def _print(*args: object) -> None:
    print(*args, flush=True)


def _format_scalar(val: float | None) -> str:
    """Format a metric value, rendering None strictly as 'undefined', never as 0."""
    if val is None:
        return "undefined"
    return f"{val:.4f}"


def _format_delta(delta: float | None) -> str:
    """Format a delta value, signed, rendering None strictly as 'undefined'."""
    if delta is None:
        return "undefined"
    return f"{delta:+.4f}"


def _render_generation(record: GenerationRecord) -> str:
    """One generation: before/after/delta/outcome, the cause it targeted, and the spec diff.

    Deltas are signed and printed as measured -- positive, negative, or zero --
    and a reverted generation is labeled as reverted, never dressed up as a
    partial win. Nothing here assumes the delta is positive or that every
    generation looks like the last one. Undefined metrics render as 'undefined',
    never as 0.
    """
    outcome = "ACCEPTED" if record.accepted else "REVERTED"
    cause = record.diagnosis.dominant_cause or record.mutation.motivating_cause
    cause_label = cause.value if hasattr(cause, "value") else str(cause) if cause is not None else "(no failures diagnosed)"
    before_str = _format_scalar(record.verdict.before)
    after_str = _format_scalar(record.verdict.after)
    delta_str = _format_delta(record.verdict.delta)
    lines = [
        f"--- generation {record.generation}: {outcome} ---",
        f"  dominant cause targeted : {cause_label}",
        f"  mutation                : {record.mutation.kind.value} ({record.mutation.target_path})",
        f"  rationale               : {record.mutation.rationale}",
        f"  before -> after         : {before_str} -> {after_str}  (delta {delta_str})",
        f"  verdict                 : {record.verdict.reason}",
    ]
    diff = diff_agent_specs(record.spec_before, record.spec_after)
    if diff:
        lines.append("  spec diff:")
        lines.extend(f"    {line}" for line in diff.rstrip("\n").splitlines())
    else:
        lines.append("  spec diff: (no textual change)")
    return "\n".join(lines)


def _stream_generation(record: GenerationRecord) -> None:
    _print()
    _print(_render_generation(record))


def _render_lineage_summary(report: LineageReport) -> str:
    lines = ["", "=== lineage summary ===", *report.summary_lines()]
    accepted = len(report.accepted_mutations)
    lines.append(f"{accepted}/{len(report.generations)} generations accepted")
    if report.noise_floor is not None:
        lines.append(
            f"keep threshold: {report.noise_floor:.6f} (measured run-to-run noise floor "
            "on the root spec, repeats=3 -- not a fixed epsilon)"
        )
    return "\n".join(lines)


def render_saved_lineage(data_or_path: dict[str, Any] | str | Path) -> str:
    """Render a saved lineage artifact (e.g. artifacts/scripted_decision_lineage.json) without running anything.

    Honors all guards:
    - Never renders an undefined metric as 0; undefined renders as 'undefined'.
    - Negative deltas, sub-noise reverts, and accepted mutations render legibly.
    - Dominant cause, mutation kind, target path, verdict reason, and spec diff are shown.
    """
    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"lineage artifact not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = data_or_path

    lines: list[str] = []

    domain = data.get("domain")
    task_count = data.get("task_count")
    backend = data.get("backend")
    label = data.get("label")

    header_parts: list[str] = []
    if domain:
        task_str = f" ({task_count} tasks)" if task_count is not None else ""
        header_parts.append(f"domain: {domain}{task_str}")
    if backend:
        header_parts.append(f"backend: {backend}")
    if header_parts:
        lines.append("  ".join(header_parts))

    if label:
        lines.append(f"label: {label}")
    if data.get("selection_policy"):
        lines.append(f"selection policy: {data['selection_policy']}")
    if data.get("root_spec_id") and data.get("final_spec_id"):
        lines.append(f"root spec: {data['root_spec_id']} -> final spec: {data['final_spec_id']}")

    lineage = data.get("lineage", [])
    for gen in lineage:
        generation = gen.get("generation", 0)
        decision = str(gen.get("decision", "unknown")).upper()
        cause = gen.get("motivating_cause") or gen.get("dominant_cause") or "(no failures diagnosed)"
        mutation_kind = gen.get("mutation_kind", "unknown")
        target_path = gen.get("target_path", "")
        mutation_label = f"{mutation_kind} ({target_path})" if target_path else mutation_kind
        rationale = gen.get("rationale")
        before = gen.get("before")
        after = gen.get("after")
        delta = gen.get("delta")
        if delta is None and before is not None and after is not None:
            delta = after - before
        reason = gen.get("reason", "")

        before_str = _format_scalar(before)
        after_str = _format_scalar(after)
        delta_str = _format_delta(delta)

        lines.append("")
        lines.append(f"--- generation {generation}: {decision} ---")
        lines.append(f"  dominant cause targeted : {cause}")
        lines.append(f"  mutation                : {mutation_label}")
        if rationale:
            lines.append(f"  rationale               : {rationale}")
        lines.append(f"  before -> after         : {before_str} -> {after_str}  (delta {delta_str})")
        if reason:
            lines.append(f"  verdict                 : {reason}")

        # Spec diff if specs are included in the artifact
        spec_before = gen.get("spec_before")
        spec_after = gen.get("spec_after")
        spec_diff = gen.get("spec_diff")
        if spec_before is not None and spec_after is not None:
            try:
                sb = AgentSpec.model_validate(spec_before) if isinstance(spec_before, dict) else spec_before
                sa = AgentSpec.model_validate(spec_after) if isinstance(spec_after, dict) else spec_after
                diff = diff_agent_specs(sb, sa)
                if diff:
                    lines.append("  spec diff:")
                    lines.extend(f"    {line}" for line in diff.rstrip("\n").splitlines())
                else:
                    lines.append("  spec diff: (no textual change)")
            except Exception:
                lines.append("  spec diff: (unable to parse specs from artifact)")
        elif spec_diff:
            lines.append("  spec diff:")
            lines.extend(f"    {line}" for line in str(spec_diff).rstrip("\n").splitlines())
        else:
            lines.append("  spec diff: (spec text not stored in artifact)")

    lines.append("")
    lines.append("=== lineage summary ===")
    accepted_count = sum(1 for g in lineage if str(g.get("decision", "")).lower() == "accepted")
    for g in lineage:
        dec = str(g.get("decision", "unknown")).lower()
        c = g.get("motivating_cause") or g.get("dominant_cause") or "(no cause)"
        m_kind = g.get("mutation_kind", "")
        t_path = g.get("target_path", "")
        m_str = f"{m_kind} ({t_path})" if t_path else m_kind
        b_val = g.get("before")
        a_val = g.get("after")
        d_val = g.get("delta")
        if d_val is None and b_val is not None and a_val is not None:
            d_val = a_val - b_val
        b_s = _format_scalar(b_val)
        a_s = _format_scalar(a_val)
        d_s = _format_delta(d_val)
        lines.append(
            f"gen {g.get('generation', 0)}: {c} -> {m_str} | "
            f"{b_s} -> {a_s} (delta {d_s}) | {dec}"
        )

    lines.append(f"{accepted_count}/{len(lineage)} generations accepted")

    noise_floor = data.get("noise_floor")
    if isinstance(noise_floor, dict):
        std = noise_floor.get("reliability_std")
        source = noise_floor.get("source", "measured noise floor")
        if std is not None:
            lines.append(f"keep threshold: {std:.6f} ({source})")
    elif isinstance(noise_floor, (int, float)):
        lines.append(
            f"keep threshold: {noise_floor:.6f} (measured run-to-run noise floor "
            "on the root spec, repeats=3 -- not a fixed epsilon)"
        )

    return "\n".join(lines)


def export_lineage_to_dict(
    report: LineageReport,
    *,
    domain: str,
    task_count: int,
    backend_name: str,
    label: str = "",
) -> dict[str, Any]:
    """Export a LineageReport to the standard JSON artifact dictionary format."""
    noise_dict = None
    if report.noise_floor is not None:
        noise_dict = {
            "source": "measured automatically by agent_engineer.loop.run_loop's own default (Evaluator, repeats=3, on the root spec)",
            "reliability_variance": report.noise_floor**2,
            "reliability_std": report.noise_floor,
        }
    return {
        "label": label or f"Lineage run on domain {domain}",
        "backend": backend_name,
        "domain": domain,
        "task_count": task_count,
        "noise_floor": noise_dict,
        "selection_policy": "run_loop default (noise-aware keep threshold)",
        "root_spec_id": report.root_spec.spec_id,
        "final_spec_id": report.final_spec.spec_id,
        "lineage": [
            {
                "generation": record.generation,
                "motivating_cause": (record.diagnosis.dominant_cause or record.mutation.motivating_cause).value,
                "mutation_kind": record.mutation.kind.value,
                "target_path": record.mutation.target_path,
                "rationale": record.mutation.rationale,
                "before": record.verdict.before,
                "after": record.verdict.after,
                "delta": record.verdict.delta,
                "decision": record.verdict.decision.value,
                "reason": record.verdict.reason,
                "spec_before": record.spec_before.model_dump(mode="json"),
                "spec_after": record.spec_after.model_dump(mode="json"),
            }
            for record in report.generations
        ],
    }


def run_domain(
    domain: str,
    *,
    max_generations: int = 3,
    metric: str = "mean_score",
    backend_name: str = "scripted",
    repeats: int = 3,
    stream: bool = True,
    output_path: str | Path | None = None,
) -> LineageReport:
    if domain not in DOMAIN_NAMES:
        raise SystemExit(
            f"unknown domain {domain!r}; registered domains: {', '.join(DOMAIN_NAMES)}"
        )
    suite = get_suite(domain)
    evaluator = get_evaluator(domain)
    tool_runtime = NullToolRuntime()
    backend = AnthropicBackend() if backend_name == "anthropic" else ScriptedBackend()

    if stream:
        _print(f"domain: {domain}  ({len(suite.tasks)} tasks)  backend: {backend_name}  metric: {metric}")
        _print("running the five-stage loop...")

    try:
        report = run_loop(
            spec_id=f"{domain}-agent",
            goal=f"Solve every task in the {domain!r} suite.",
            tool_runtime=tool_runtime,
            backend=backend,
            evaluator=evaluator,
            evaluator_id=f"{domain}.cli-run",
            task_suite=suite,
            max_generations=max_generations,
            metric=metric,
            on_generation=_stream_generation if stream else None,
        )
    except LoopStalled as error:
        raise SystemExit(f"loop stalled: {error}") from error

    if stream:
        _print(_render_lineage_summary(report))

    if output_path is not None:
        out_dict = export_lineage_to_dict(
            report,
            domain=domain,
            task_count=len(suite.tasks),
            backend_name=backend_name,
        )
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
        if stream:
            _print(f"\nsaved lineage artifact to {out_p}")

    # Results presentation, entirely through agent_engineer.evaluation.report: the
    # engine's own pass_rate/mean_score bookkeeping decided the lineage above,
    # but the numbers shown to a human come from the measurement harness, with
    # its guards (no accuracy without cost, no improvement without a before
    # number, undefined never rendered as zero) intact.
    runner = TrajectoryRunner(backend=backend, tool_runtime=tool_runtime, evaluator=evaluator)
    task_runner = runner.as_task_runner()
    measurement = Evaluator([suite], evaluators={domain: evaluator}, repeats=repeats)
    baseline_report = measurement.run_iteration(0, report.root_spec, task_runner)
    final_report = measurement.run_iteration(1, report.final_spec, task_runner)

    _print()
    _print("=== results (agent_engineer.evaluation.report) ===")
    _print(render_report([baseline_report, final_report]))

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-engineer",
        description="Run the agent-engineer loop against any registered domain, or render saved lineages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list the registered domain names")

    run_parser = sub.add_parser("run", help="run the loop against one domain")
    run_parser.add_argument("domain", choices=DOMAIN_NAMES, help="registered domain name")
    run_parser.add_argument(
        "--max-generations", type=int, default=3, help="mutation attempts to allow (default: 3)"
    )
    run_parser.add_argument(
        "--metric",
        choices=("pass_rate", "mean_score"),
        default="mean_score",
        help="which scalar stage 5 compares before/after a mutation (default: mean_score)",
    )
    run_parser.add_argument(
        "--backend",
        choices=("scripted", "anthropic"),
        default="scripted",
        help="scripted: deterministic stand-in, no API key needed. "
        "anthropic: a real model, requires ANTHROPIC_API_KEY (default: scripted)",
    )
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="repeat runs per task for the results table's reliability metric (default: 3, minimum 3)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="save the resulting lineage artifact to a JSON file",
    )
    run_parser.add_argument(
        "--quiet", action="store_true", help="suppress streaming output; only print the final report"
    )

    for cmd in ("render", "show"):
        render_parser = sub.add_parser(cmd, help="render a saved lineage artifact from JSON")
        render_parser.add_argument("artifact_path", type=str, help="path to JSON lineage artifact")

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        args_list = list(argv)
    else:
        args_list = sys.argv[1:]

    # Ergonomic shortcuts:
    # 1. python -m agent_engineer code_math -> python -m agent_engineer run code_math
    # 2. python -m agent_engineer artifacts/scripted_decision_lineage.json -> python -m agent_engineer render ...
    if args_list and not args_list[0].startswith("-"):
        first = args_list[0]
        if first in DOMAIN_NAMES:
            args_list = ["run", *args_list]
        elif first.endswith(".json") or (len(args_list) == 1 and os.path.isfile(first)):
            args_list = ["render", *args_list]

    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.command == "list":
        for name in DOMAIN_NAMES:
            _print(name)
        return 0

    if args.command == "run":
        run_domain(
            args.domain,
            max_generations=args.max_generations,
            metric=args.metric,
            backend_name=args.backend,
            repeats=args.repeats,
            stream=not args.quiet,
            output_path=args.output,
        )
        return 0

    if args.command in ("render", "show"):
        rendered = render_saved_lineage(args.artifact_path)
        _print(rendered)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

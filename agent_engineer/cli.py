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
import os
import re
import sys
from typing import Any

from agent_engineer.domains import DOMAIN_NAMES, get_evaluator, get_suite
from agent_engineer.evaluation import Evaluator
from agent_engineer.evaluation.report import render_report
from agent_engineer.loop import GenerationRecord, LineageReport, LoopStalled, run_loop
from agent_engineer.ports import ToolResult, ToolSchema
from agent_engineer.schemas import diff_agent_specs
from agent_engineer.stages.evaluate import TrajectoryRunner

__all__ = ["main", "NullToolRuntime", "ScriptedBackend", "AnthropicBackend"]


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


def _render_generation(record: GenerationRecord) -> str:
    """One generation: before/after/delta/outcome, the cause it targeted, and the spec diff.

    Deltas are signed and printed as measured -- positive, negative, or zero --
    and a reverted generation is labeled as reverted, never dressed up as a
    partial win. Nothing here assumes the delta is positive or that every
    generation looks like the last one.
    """
    outcome = "ACCEPTED" if record.accepted else "REVERTED"
    cause = record.diagnosis.dominant_cause
    cause_label = cause.value if cause is not None else "(no failures diagnosed)"
    lines = [
        f"--- generation {record.generation}: {outcome} ---",
        f"  dominant cause targeted : {cause_label}",
        f"  mutation                : {record.mutation.kind.value} ({record.mutation.target_path})",
        f"  rationale               : {record.mutation.rationale}",
        f"  before -> after         : {record.verdict.before:.4f} -> {record.verdict.after:.4f}"
        f"  (delta {record.verdict.delta:+.4f})",
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


# --------------------------------------------------------------------------- #
# The run command
# --------------------------------------------------------------------------- #


def run_domain(
    domain: str,
    *,
    max_generations: int = 3,
    metric: str = "mean_score",
    backend_name: str = "scripted",
    repeats: int = 3,
    stream: bool = True,
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
        description="Run the agent-engineer loop against any registered domain, by name.",
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
        "--quiet", action="store_true", help="suppress streaming output; only print the final report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

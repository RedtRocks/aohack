"""Render artifacts/code_math_gpt5_nano_lineage.json as a human-readable Markdown
report the demo can display and the write-up can quote verbatim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
JSON_PATH = ARTIFACT_DIR / "code_math_gpt5_nano_lineage.json"
MD_PATH = ARTIFACT_DIR / "code_math_gpt5_nano_lineage.md"


def _fmt_metric(metric: dict, unit: str = "") -> str:
    if metric["value"] is None:
        return f"undefined ({metric['reason']})"
    return f"{metric['value']:.4f}{unit} (n={metric['sample_size']})"


def render(data: dict) -> str:
    lines = [
        f"# {data['domain']} x a real model: a live run through the engine loop",
        "",
        f"Primary model: `{data['primary_model']}` &middot; fallback: `{data['fallback_model']}` "
        f"&middot; domain: `{data['domain']}` &middot; {data['task_count']} tasks &middot; "
        f"**{data['real_model_calls_made']} real API calls made** "
        f"({data['primary_calls']} primary / {data['fallback_calls']} fallback)",
        "",
        "A weak model was chosen deliberately, not as a fallback: a strong baseline agent "
        "leaves the loop no headroom, so every mutation lands in the noise. A weak model "
        "gives a low baseline with real room to improve, and it is cheap and fast enough "
        "to run several times per task so run-to-run variance means something.",
        "",
        "## Noise floor",
        "",
        f"The root spec was run through the real suite {data['noise_floor']['replicates']} "
        "independent times before any mutation, to measure how much `mean_score` -- the "
        "metric the loop's own accept/revert decision is made on -- moves on its own, from "
        "model sampling alone.",
        "",
        f"- replicate `mean_score` values: {data['noise_floor']['replicate_scores']}",
        f"- population variance: {data['noise_floor']['population_variance']:.6f}",
        f"- population std (the noise floor): {data['noise_floor']['population_std']:.6f}",
        "",
        "Any lineage delta below this std is flagged `within_noise_floor` below and must not "
        "be read as a real improvement or regression.",
        "",
        "## Baseline vs. final (accuracy always paired with cost)",
        "",
        "| | accuracy | reliability (variance) | cost (mean tokens/run) | speed (mean seconds/run) |",
        "|---|---|---|---|---|",
        f"| baseline (`{data['root_spec_id']}`) | "
        f"{_fmt_metric(data['baseline']['accuracy'])} | "
        f"{_fmt_metric(data['baseline']['reliability'])} | "
        f"{_fmt_metric(data['baseline']['cost_tokens'])} | "
        f"{_fmt_metric(data['baseline']['speed_seconds'], 's')} |",
        f"| final (`{data['final_spec_id']}`) | "
        f"{_fmt_metric(data['final']['accuracy'])} | "
        f"{_fmt_metric(data['final']['reliability'])} | "
        f"{_fmt_metric(data['final']['cost_tokens'])} | "
        f"{_fmt_metric(data['final']['speed_seconds'], 's')} |",
        "",
        "Both rows are computed by the real, frozen `agent_engineer.evaluation.Evaluator` "
        f"harness, `repeats={data['baseline']['repeats']}`, over the exact same recorded "
        "trajectories the noise-floor measurement used -- not hand-computed here.",
        "",
        "## The real lineage, verbatim",
        "",
        "| gen | dominant cause | mutation | before | after | delta | decision | within noise floor |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in data["lineage"]:
        lines.append(
            f"| {record['generation']} | {record['motivating_cause']} | "
            f"{record['mutation_kind']} ({record['target_path']}) | "
            f"{record['before']:.4f} | {record['after']:.4f} | {record['delta']:+.4f} | "
            f"{record['decision']} | {'yes' if record['within_noise_floor'] else 'no'} |"
        )
    lines += ["", "Rationale behind each proposed mutation:", ""]
    for record in data["lineage"]:
        lines.append(f"- **gen {record['generation']}** ({record['decision']}): {record['rationale']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        if not json_path.is_absolute():
            json_path = (Path.cwd() / json_path).resolve()
        md_path = json_path.with_suffix(".md")
    else:
        json_path = JSON_PATH
        md_path = MD_PATH

    if not json_path.exists():
        print(f"no artifact at {json_path}; run experiments/run_real_model_gate.py first",
              file=sys.stderr)
        raise SystemExit(1)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_path.write_text(render(data), encoding="utf-8")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

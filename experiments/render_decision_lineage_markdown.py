"""Render artifacts/scripted_decision_lineage.json as Markdown for the demo.

Labelled SCRIPTED throughout, deliberately -- see the module docstring on
``generate_scripted_decision_lineage.py`` for why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
JSON_PATH = ARTIFACT_DIR / "scripted_decision_lineage.json"
MD_PATH = ARTIFACT_DIR / "scripted_decision_lineage.md"


def render(data: dict) -> str:
    lines = [
        "# code_math keep-or-revert, exercised end to end (SCRIPTED)",
        "",
        f"**{data['label']}.** {data['backend']}",
        "",
        f"Domain: `{data['domain']}` &middot; {data['task_count']} tasks &middot; "
        f"selection policy: `{data['selection_policy']}`",
        "",
        "## Noise floor",
        "",
        f"{data['noise_floor']['source']}.",
        "",
        f"- reliability (variance): `{data['noise_floor']['reliability_variance']:.6f}`",
        f"- noise floor (std): `{data['noise_floor']['reliability_std']:.6f}`",
        "",
        "## The lineage",
        "",
        "| gen | dominant cause | mutation | before | after | delta | decision |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in data["lineage"]:
        lines.append(
            f"| {record['generation']} | {record['motivating_cause']} | "
            f"{record['mutation_kind']} ({record['target_path']}) | "
            f"{record['before']:.4f} | {record['after']:.4f} | {record['delta']:+.4f} | "
            f"{record['decision']} |"
        )
    lines += [
        "",
        f"Root spec: `{data['root_spec_id']}` &rarr; final spec: `{data['final_spec_id']}`",
        "",
        "Reasons, verbatim from the selection stage:",
        "",
    ]
    for record in data["lineage"]:
        lines.append(f"- **gen {record['generation']}** ({record['decision']}): {record['reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not JSON_PATH.exists():
        print(f"no artifact at {JSON_PATH}; run experiments/generate_scripted_decision_lineage.py first",
              file=sys.stderr)
        raise SystemExit(1)
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    MD_PATH.write_text(render(data), encoding="utf-8")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()

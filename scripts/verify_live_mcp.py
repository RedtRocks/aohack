"""Opt-in live verification script for the mcp_everything domain.

Demonstrates the engine calling real MCP tools on the reference everything server
via stdio transport (Node.js/npx).

Usage:
    python scripts/verify_live_mcp.py
"""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agent_engineer.domains.mcp_everything import (
    get_evaluator,
    get_suite,
    get_tool_runtime,
    StdioTransport,
    MCPClient,
)
from agent_engineer.ports import AgentAction
from agent_engineer.schemas import AgentSpec
from agent_engineer.stages.evaluate import TrajectoryRunner


def main() -> int:
    print("=" * 60)
    print("LIVE MCP PROTOCOL VERIFICATION")
    print("=" * 60)

    print("1. Connecting to live MCP server via StdioTransport...")
    try:
        transport = StdioTransport()
        client = MCPClient(transport=transport)
        runtime = get_tool_runtime(client=client)
    except Exception as exc:
        print(f"FAILED to start live MCP server: {exc}")
        return 1

    try:
        print("2. Discovering tools dynamically from live server (tools/list)...")
        schemas = runtime.schemas()
        print(f"   Discovered {len(schemas)} tools dynamically:")
        for s in schemas:
            print(f"     - {s.name}: {s.description[:60] if s.description else '(no desc)'}")

        suite = get_suite()
        evaluator = get_evaluator()

        print("\n3. Exercising engine (TrajectoryRunner) over real protocol...")

        class LiveEchoBackend:
            def next_action(self, spec, task, tools, history):
                if not history:
                    return AgentAction(
                        tool_name="echo",
                        args={"message": "Hello MCP!"},
                    )
                return AgentAction(final_answer=history[0].result)

        spec = AgentSpec(
            spec_id="live-mcp-agent",
            system_prompt="You are an autonomous agent with real MCP tools.",
            tools=tuple(s.name for s in schemas),
        )

        runner = TrajectoryRunner(backend=LiveEchoBackend(), tool_runtime=runtime, evaluator=evaluator)
        task = suite.tasks[0]  # mcp-echo
        print(f"   Running task: {task.task_id} -> {task.prompt}")

        record = runner.run_task(spec, task)
        traj = record.trajectory

        print("\n4. Verification Results:")
        print(f"   Trajectory ID: {traj.trajectory_id}")
        print(f"   Tool calls made: {len(traj.tool_calls)}")
        for tc in traj.tool_calls:
            print(f"     Tool: {tc.tool_name}")
            print(f"     Args: {tc.args}")
            print(f"     Result: {tc.result}")
            print(f"     Error: {tc.error}")
        print(f"   Final Answer: {traj.final_answer}")
        print(f"   Verdict: passed={traj.verdict.passed}, score={traj.verdict.score}")
        print(f"   Rationale: {traj.verdict.rationale}")

        if traj.verdict.passed and len(traj.tool_calls) == 1 and traj.tool_calls[0].error is None:
            print("\n*** REAL MCP TOOL EXECUTION VERIFIED THROUGH THE ENGINE ***")
            return 0
        else:
            print("\nVerification did not pass expected criteria.")
            return 1
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())

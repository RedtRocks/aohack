"""ToolRuntime adapter for the MCP everything domain.

Conforms to the ToolRuntime protocol defined in agent_engineer.ports.
Discovers tool schemas dynamically from the server at runtime and translates
invocations to MCP JSON-RPC calls.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_engineer.domains.mcp_everything.client import MCPClient, MCPServerError
from agent_engineer.evaluation import TaskSpec
from agent_engineer.ports import ToolResult, ToolRuntime, ToolSchema

logger = logging.getLogger(__name__)


class MCPToolRuntime:
    """Adapts an MCPClient to the engine's ToolRuntime protocol."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or MCPClient()

    @property
    def client(self) -> MCPClient:
        return self._client

    def schemas(self) -> tuple[ToolSchema, ...]:
        """Query the MCP server dynamically for its available tools."""
        try:
            tools = self._client.list_tools()
            return tuple(
                ToolSchema(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                )
                for t in tools
            )
        except MCPServerError as exc:
            logger.warning("Failed to list tools from MCP server: %s", exc)
            return ()

    def invoke(self, task: TaskSpec, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Call an MCP tool over the real protocol."""
        try:
            resp = self._client.call_tool(tool_name, args)
            if "error" in resp:
                err_data = resp["error"]
                code = err_data.get("code")
                msg = err_data.get("message")
                return ToolResult(error=f"MCP error {code}: {msg}")

            result = resp.get("result", {})
            if result.get("isError"):
                content = result.get("content", [])
                err_text = " ".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
                return ToolResult(error=err_text or "MCP tool reported an error")

            content = result.get("content", [])
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            if texts:
                return ToolResult(value="\n".join(texts))
            return ToolResult(value=content)
        except MCPServerError as exc:
            return ToolResult(error=f"MCP communication failed: {exc}")
        except Exception as exc:
            return ToolResult(error=f"Unexpected error invoking tool {tool_name!r}: {exc}")

    def close(self) -> None:
        self._client.close()

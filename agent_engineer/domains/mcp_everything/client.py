"""MCP client connecting to an MCP server using JSON-RPC 2.0 over stdio transport.

Provides dynamic tool discovery and tool execution adhering to the Model
Context Protocol (2024-11-05). Supports real subprocess stdio communication
and offline stubbed transports for test environments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MCPServerError(Exception):
    """Raised when communication with the MCP server fails."""


class MCPServerNotFoundError(MCPServerError):
    """Raised when the MCP server executable cannot be found or started."""


class MCPTransport(Protocol):
    """Protocol for exchanging JSON-RPC messages with an MCP server."""

    def send(self, message: dict[str, Any]) -> None: ...
    def receive(self, timeout: float = 10.0) -> dict[str, Any] | None: ...
    def close(self) -> None: ...


class StdioTransport:
    """Stdio transport running the MCP server process locally."""

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or self._resolve_default_command()
        self._proc: subprocess.Popen[str] | None = None
        self._start()

    @staticmethod
    def _resolve_default_command() -> list[str]:
        """Find the command to launch the everything server."""
        standalone = shutil.which("mcp-server-everything")
        if standalone:
            return [standalone]
        npx = shutil.which("npx")
        if npx:
            return [npx, "-y", "@modelcontextprotocol/server-everything"]
        raise MCPServerNotFoundError(
            "Neither 'mcp-server-everything' nor 'npx' executable was found on PATH. "
            "Please ensure Node.js and npx are installed to run the real MCP server."
        )

    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise MCPServerNotFoundError(
                f"Failed to start MCP server process {self._command!r}: {exc}"
            ) from exc

    def send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise MCPServerError("MCP server process is not running")
        payload = json.dumps(message) + "\n"
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except OSError as exc:
            raise MCPServerError(f"Failed to write to MCP server stdin: {exc}") from exc

    def receive(self, timeout: float = 10.0) -> dict[str, Any] | None:
        if self._proc is None or self._proc.stdout is None:
            raise MCPServerError("MCP server process is not running")

        # In standard stdio operation, readline blocks until a newline arrives or process exits
        while True:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    return None
                continue
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON output from MCP server: %s", line)
                continue

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None


class StubTransport:
    """In-memory stubbed transport for offline testing without network or Node.js."""

    def __init__(self, fixtures_path: str | Path | None = None) -> None:
        self._fixtures: dict[str, Any] = {}
        target_path = Path(fixtures_path) if fixtures_path else Path(__file__).parent / "fixtures.json"
        if target_path.is_file():
            try:
                self._fixtures = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self._queue: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        msg_id = message.get("id")

        if method == "initialize":
            init_data = self._fixtures.get("init", {}).get("result") or {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "mcp-servers/everything", "version": "2.0.0"},
            }
            self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": init_data})
        elif method == "notifications/initialized":
            pass  # No response expected for notifications
        elif method == "tools/list":
            tools_data = self._fixtures.get("tools", {}).get("result") or {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echoes back the input string",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    },
                    {
                        "name": "get-sum",
                        "description": "Returns the sum of two numbers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                ]
            }
            self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": tools_data})
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name", "")
            args = params.get("arguments", {})

            if name == "echo":
                msg = args.get("message", "")
                res = {"content": [{"type": "text", "text": f"Echo: {msg}"}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            elif name in ("get-sum", "add"):
                a = args.get("a", 0)
                b = args.get("b", 0)
                res = {"content": [{"type": "text", "text": f"The sum of {a} and {b} is {a + b}."}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            elif name in ("get-tiny-image",):
                res = {"content": [{"type": "image", "data": "iVBORw0KGgoAAA==", "mimeType": "image/png"}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            elif name in ("get-annotated-message", "annotatedMessage"):
                m_type = args.get("messageType", "info")
                res = {"content": [{"type": "text", "text": f"Annotated {m_type} message"}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            elif name in ("get-env", "printEnv"):
                res = {"content": [{"type": "text", "text": "PATH=/usr/bin:/bin\nNODE_ENV=test"}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            elif name in ("trigger-long-running-operation", "longRunningOperation"):
                res = {"content": [{"type": "text", "text": "Long running operation completed"}]}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
            else:
                res = {"content": [{"type": "text", "text": f"MCP error -32602: Tool {name} not found"}], "isError": True}
                self._queue.append({"jsonrpc": "2.0", "id": msg_id, "result": res})
        else:
            self._queue.append({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })

    def receive(self, timeout: float = 10.0) -> dict[str, Any] | None:
        if self._queue:
            return self._queue.pop(0)
        return None

    def close(self) -> None:
        self._queue.clear()


class MCPClient:
    """Client for the Model Context Protocol speaking JSON-RPC 2.0."""

    def __init__(self, transport: MCPTransport | None = None) -> None:
        self._transport = transport
        self._next_id = 1
        self._initialized = False

    def _get_next_id(self) -> int:
        cur = self._next_id
        self._next_id += 1
        return cur

    def start(self) -> None:
        """Start the transport and perform the protocol initialization handshake."""
        if self._transport is None:
            self._transport = StdioTransport()
        if not self._initialized:
            self._handshake()

    def _handshake(self) -> None:
        assert self._transport is not None
        init_id = self._get_next_id()
        init_req = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agent-engineer-mcp-client",
                    "version": "0.1.0",
                },
            },
        }
        resp = self._request(init_req)
        if "error" in resp:
            raise MCPServerError(f"Initialization failed: {resp['error']}")
        # Send initialized notification per protocol spec
        self._transport.send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        self._initialized = True

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._transport is None:
            raise MCPServerError("Transport is not connected")
        req_id = message.get("id")
        self._transport.send(message)
        while True:
            resp = self._transport.receive()
            if resp is None:
                raise MCPServerError("Connection closed before response received")
            # Filter responses matching request id
            if resp.get("id") == req_id:
                return resp
            logger.debug("Received out-of-band message: %s", resp)

    def list_tools(self) -> list[dict[str, Any]]:
        """Discover tools dynamically from the server at runtime.

        Never hardcodes schemas; issues tools/list to query the live MCP server.
        """
        if not self._initialized:
            self.start()
        req_id = self._get_next_id()
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/list",
            "params": {},
        }
        resp = self._request(req)
        if "error" in resp:
            raise MCPServerError(f"Failed to list tools: {resp['error']}")
        result = resp.get("result", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool on the server over the real MCP protocol."""
        if not self._initialized:
            self.start()
        req_id = self._get_next_id()
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        return self._request(req)

    def close(self) -> None:
        """Close the underlying transport."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._initialized = False

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

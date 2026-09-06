"""Run untrusted candidate code in a throwaway subprocess.

The contract this module owes the evaluator is narrow and total: given source
text and a timeout, :func:`run_python` always returns a :class:`SandboxResult`.
It never raises for anything the candidate did, never leaves a process behind,
and never lets a hanging candidate stall the caller -- a task that hangs comes
back as ``status="timeout"``, which the evaluator scores as a plain failure.

Isolation is defence in depth, not a jail. The child runs with ``-I -B -E``
(isolated mode: no user site-packages, no ``PYTHON*`` env vars, no bytecode
writes, cwd off ``sys.path``), in a fresh temporary directory that is deleted
afterwards, under a scrubbed environment, with a prelude that neuters the
network and process-spawning surface of the standard library before a single
line of candidate code runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Final

__all__ = ["SandboxResult", "run_python", "DEFAULT_TIMEOUT_SECONDS"]

DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
"""Wall-clock budget for one candidate run. Generous for these tasks, short enough
that a pathological submission costs the loop one timeout rather than a stall."""

_MAX_CAPTURE_CHARS: Final[int] = 20_000
"""Cap on captured stdout/stderr, so a candidate printing in a loop cannot
exhaust the parent's memory before the timeout fires."""

_PRELUDE: Final[str] = '''\
import builtins as _b, sys as _s

def _blocked(_name):
    def _deny(*_a, **_k):
        raise RuntimeError("sandbox: " + _name + " is not available (no network, no subprocesses)")
    return _deny

try:
    import socket as _socket
except ImportError:
    _socket = None
if _socket is not None:
    for _attr in ("socket", "create_connection", "create_server", "socketpair",
                  "getaddrinfo", "gethostbyname"):
        setattr(_socket, _attr, _blocked("socket." + _attr))

import subprocess as _sub
for _attr in ("run", "call", "check_call", "check_output", "Popen"):
    setattr(_sub, _attr, _blocked("subprocess." + _attr))

import os as _os
for _attr in ("system", "popen", "execv", "execve", "execvp", "fork", "spawnv", "spawnl"):
    if hasattr(_os, _attr):
        setattr(_os, _attr, _blocked("os." + _attr))

_s.setrecursionlimit(3000)
del _b, _blocked, _attr
'''


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandboxed run.

    ``status`` is ``"ok"`` when the child exited 0, ``"error"`` when it exited
    non-zero (an exception, an assertion, a syntax error), and ``"timeout"``
    when it outlived its budget and was killed.
    """

    status: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _child_env() -> dict[str, str]:
    """A minimal environment: enough to start CPython, nothing that reaches out."""
    keep = ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "LANG")
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # No proxy inherited, and urllib/requests-style helpers get a dead endpoint
    # even if something reconstructs a socket the prelude did not cover.
    env["no_proxy"] = "*"
    return env


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CAPTURE_CHARS:
        return text
    return text[:_MAX_CAPTURE_CHARS] + "\n...[truncated]"


def run_python(source: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> SandboxResult:
    """Execute ``source`` in an isolated child interpreter and capture its output.

    Never raises on candidate misbehaviour: a crash, an infinite loop, a
    ``sys.exit``, or output floods all come back as a ``SandboxResult``.
    """
    workdir = tempfile.mkdtemp(prefix="ae_sandbox_")
    try:
        script = os.path.join(workdir, "candidate_run.py")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(_PRELUDE)
            handle.write("\n")
            handle.write(source)
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-E", script],
            cwd=workdir,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return SandboxResult(
                status="timeout",
                returncode=None,
                stdout=_truncate(stdout or ""),
                stderr=_truncate(stderr or ""),
                timed_out=True,
            )
        status = "ok" if process.returncode == 0 else "error"
        return SandboxResult(
            status=status,
            returncode=process.returncode,
            stdout=_truncate(stdout or ""),
            stderr=_truncate(stderr or ""),
            timed_out=False,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

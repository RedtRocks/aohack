"""Thin :class:`~agent_engineer.ports.ModelBackend` implementations over real
models, via OpenAI-compatible chat-completions APIs, plus a fallback wrapper
that tries a primary provider first.

This is a standalone experiment module, not part of the installable
``agent_engineer`` package: it exists to answer one question -- does the
engine's loop actually improve a real agent, not just wire together correctly
against a scripted stand-in -- and it is intentionally kept outside the engine
so nothing here can be mistaken for engine code.

**Strategy is real, not decorative.** Earlier versions of this module made one
call per task regardless of ``spec.strategy``, which made ``strategy_changed``
mutations a guaranteed no-op: reordering the loop shape could never move the
score because nothing here ever read it. That silently invalidated a third of
the mutation vocabulary against these text-only domains (no tools, so
``stopping_adjusted`` is separately and legitimately a no-op here: a step
budget only binds when there is more than one step, and there never is
without a tool call). Two strategies now genuinely change what is sent to the
model, each as two real provider calls rather than one:

* ``PLAN_THEN_EXECUTE`` -- a first call asks only for a short plan, withheld
  from the model as an answer; a second call hands that plan back as context
  and asks for the real final answer. This is the literal "explicit step for
  the deliberation it is currently skipping" the mutation ladder's own
  rationale claims.
* ``REFLEXION`` -- a first call drafts an answer; a second call shows the
  draft back alongside the task and required format and asks the model to
  either confirm it or emit a corrected final answer. This is a genuine
  retry-on-format-failure, which is what a ``REFLEXION`` escalation is
  supposed to buy an ``output_format_violation`` cause.

Every other strategy (``SINGLE_SHOT``, ``REACT``, ``TREE_SEARCH``,
``DELEGATING_SUBAGENTS``) makes one HTTP call per *step*, not per task: when
``tools`` is non-empty the request carries an OpenAI-style ``tools`` array
and ``history`` (the ``ToolCall``s made so far) is replayed as prior
assistant/tool messages, so the model can make a dependent call and see its
result on the next step -- otherwise a domain like ``api_orchestration``
could never be exercised through its tools at all. When ``tools`` is empty
(code_math, extraction) the request carries no ``tools`` field and the model
always answers on the first turn, exactly as before -- the tool-calling path
is a strict superset of the old single-shot behavior, not a replacement for
it. ``PLAN_THEN_EXECUTE`` and ``REFLEXION`` do not thread ``tools`` or
``history`` through their two internal calls: they exist for the text-only
domains this experiment was built against, and combining two-call
deliberation with a tool-calling ReAct loop is a distinct escalation nothing
here claims to model. ``spec.memory`` is still not read here: memory effects
are the engine's own concern (episodic retrieval, wired through
``TrajectoryRunner``'s ``memory_store``), not this backend's.
* No domain vocabulary anywhere in this file. The system prompt comes from
  the engine's own :class:`~agent_engineer.stages.synthesize.TemplateSynthesizer`
  and the mutation ladder's own guidance text; this module only transports
  those strings to the API and the API's reply back.
* API keys are read from environment variables at call time, never
  hardcoded, never logged, and never included in any exception message or
  request log -- only the auth header carries one, and nothing here prints
  headers.

Two providers are wired:

* :class:`TensorMuxGLMBackend` -- GLM 4.7 Flash via TensorMux, the primary.
  Reads ``TENSORMUX_API_KEY``.
* :class:`GPT5NanoBackend` -- GPT-5 nano via the standard OpenAI API, the
  fallback. Reads ``OPENAI_API_KEY``. ``reasoning_effort: "minimal"`` is
  requested explicitly: GPT-5 nano defaults to spending hidden reasoning
  tokens even on trivial replies (confirmed by a smoke call: 74 completion
  tokens, 64 of them reasoning, for the one-word reply "OK"); minimal effort
  keeps each call fast and cheap without touching what the model is asked.

:class:`FallbackBackend` tries the primary for every call and only falls back
to the secondary once the primary has exhausted its own retries -- for a
provider outage or a rate limit, not for an occasional blip.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from agent_engineer.ports import AgentAction
from agent_engineer.schemas import OrchestrationStrategy

TIMEOUT_SECONDS = 60
MAX_RETRIES = 1
"""One retry maximum -- a call either works on the second try or the whole run
stops and reports, rather than looping on a provider that is down."""

DEFAULT_MAX_TOKENS = 400
"""code_math answers are short (a number, or one short function). Capped
tightly so a model that rambles cannot silently multiply the bill."""


class ApiKeyMissing(RuntimeError):
    """Raised when a required API key env var is not set. Never simulated around."""


def _render_tools(tools) -> list[dict]:
    """``ToolSchema`` tuple -> the OpenAI ``tools`` array shape. Empty in, empty out."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _simple_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """The plain two-message shape, with no tool history -- what
    ``PLAN_THEN_EXECUTE`` and ``REFLEXION`` send for each of their two
    internal calls, since neither threads tools through this backend."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_messages(system_prompt: str, user_prompt: str, history) -> list[dict]:
    """Replay ``history`` (the ``ToolCall``s made so far) as prior turns, since
    this backend is stateless across HTTP calls: every step reconstructs the
    whole conversation from the trajectory recorded outside it, rather than
    holding any state of its own."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for call in history:
        call_id = f"call_{call.ordinal}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(call.args),
                        },
                    }
                ],
            }
        )
        content = call.error if call.error is not None else json.dumps(call.result)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return messages


class OpenAICompatibleBackend:
    """Calls a real model once per task over an OpenAI-compatible chat-completions
    API. Stateless across calls, no memoization: every call is an independent
    sample from the model, which is what lets a later step measure real
    run-to-run variance rather than a cached echo.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        reasoning_effort: str | None = None,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        key = os.environ.get(api_key_env)
        if not key:
            raise ApiKeyMissing(f"{api_key_env} is not set in the environment")
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._disable_thinking = disable_thinking
        self._max_tokens = max_tokens
        self.calls_made = 0
        self.total_tokens = 0

    def next_action(self, spec, task, tools, history) -> AgentAction:
        if spec.strategy is OrchestrationStrategy.PLAN_THEN_EXECUTE:
            return self._plan_then_execute(spec.system_prompt, task.prompt)
        if spec.strategy is OrchestrationStrategy.REFLEXION:
            return self._reflexion(spec.system_prompt, task.prompt)
        messages = _build_messages(spec.system_prompt, task.prompt, history)
        api_tools = _render_tools(tools)
        return self._call_with_retry(messages, api_tools)

    def _call_with_retry(self, messages: list[dict], api_tools: list[dict]) -> AgentAction:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._call(messages, api_tools)
            except Exception as error:  # transient network/API failure
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _plan_then_execute(self, system_prompt: str, task_prompt: str) -> AgentAction:
        """Two real calls: a withheld plan, then the answer written with it in hand."""
        plan_prompt = (
            f"{task_prompt}\n\n"
            "Do not answer yet. First write a short, numbered plan for how you will "
            "solve this. Output only the plan."
        )
        plan_action = self._call_with_retry(_simple_messages(system_prompt, plan_prompt), [])
        answer_prompt = (
            f"{task_prompt}\n\n"
            "You already wrote this plan for solving it:\n"
            f"{plan_action.final_answer}\n\n"
            "Now carry out that plan and give the final answer, in the exact format "
            "the task requires. Output only the final answer."
        )
        answer_action = self._call_with_retry(_simple_messages(system_prompt, answer_prompt), [])
        return AgentAction(
            final_answer=answer_action.final_answer,
            prompt_tokens=plan_action.prompt_tokens + answer_action.prompt_tokens,
            completion_tokens=plan_action.completion_tokens + answer_action.completion_tokens,
        )

    def _reflexion(self, system_prompt: str, task_prompt: str) -> AgentAction:
        """Two real calls: a draft answer, then an explicit self-check-and-revise pass.

        This is the retry-on-format-failure a REFLEXION escalation is supposed to
        buy an ``output_format_violation`` cause: the second call is shown its own
        draft and the task again, and can either confirm it or replace it.
        """
        draft_action = self._call_with_retry(_simple_messages(system_prompt, task_prompt), [])
        revise_prompt = (
            f"{task_prompt}\n\n"
            "You drafted this answer:\n"
            f"{draft_action.final_answer}\n\n"
            "Check it against the task above: is every fact correct, and does the "
            "answer's shape exactly match what was asked (no extra words, no wrong "
            "structure)? If yes, repeat it unchanged. If not, output the corrected "
            "final answer instead. Output only the final answer, nothing else."
        )
        revised_action = self._call_with_retry(_simple_messages(system_prompt, revise_prompt), [])
        return AgentAction(
            final_answer=revised_action.final_answer,
            prompt_tokens=draft_action.prompt_tokens + revised_action.prompt_tokens,
            completion_tokens=draft_action.completion_tokens + revised_action.completion_tokens,
        )

    def _call(self, messages: list[dict], api_tools: list[dict]) -> AgentAction:
        self.calls_made += 1
        token_key = "max_completion_tokens" if self._reasoning_effort else "max_tokens"
        payload = {
            "model": self._model,
            token_key: self._max_tokens,
            "messages": messages,
        }
        if api_tools:
            payload["tools"] = api_tools
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if self._disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]["message"]
        usage = body.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        self.total_tokens += prompt_tokens + completion_tokens

        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            function = call.get("function", {})
            name = function.get("name")
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (TypeError, ValueError):
                args = {}
            return AgentAction(
                tool_name=name,
                args=args if isinstance(args, dict) else {},
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return AgentAction(
            final_answer=choice.get("content") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class GPT5NanoBackend(OpenAICompatibleBackend):
    """GPT-5 nano over the standard OpenAI API. Reads ``OPENAI_API_KEY``."""

    def __init__(self, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        super().__init__(
            base_url="https://api.openai.com/v1",
            model="gpt-5-nano",
            api_key_env="OPENAI_API_KEY",
            reasoning_effort="minimal",
            max_tokens=max_tokens,
        )


class TensorMuxGLMBackend(OpenAICompatibleBackend):
    """GLM 4.7 Flash via TensorMux's OpenAI-compatible endpoint.
    Reads ``TENSORMUX_API_KEY``."""

    def __init__(self, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        super().__init__(
            base_url="https://api.tensormux.com/v1",
            model="glm-4-7-flash",
            api_key_env="TENSORMUX_API_KEY",
            # GLM 4.7 Flash defaults to a hidden reasoning trace before any visible
            # content -- confirmed by a smoke call: max_tokens=400 was entirely
            # consumed by reasoning, finish_reason "length", content null. Disabling
            # it is a ~7x cost cut AND the difference between getting an answer at
            # all and getting truncated silence; it is a universal generation
            # parameter, not a per-task or per-domain tweak.
            disable_thinking=True,
            max_tokens=max_tokens,
        )


class FallbackBackend:
    """Tries ``primary`` for every call; only tries ``secondary`` once ``primary``
    has exhausted its own retries (an outage or a rate limit), not for an
    occasional blip. Reports which provider actually answered each call."""

    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary
        self.primary_calls = 0
        self.secondary_calls = 0

    @property
    def calls_made(self) -> int:
        return getattr(self._primary, "calls_made", 0) + getattr(self._secondary, "calls_made", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._primary, "total_tokens", 0) + getattr(self._secondary, "total_tokens", 0)

    def next_action(self, spec, task, tools, history) -> AgentAction:
        try:
            action = self._primary.next_action(spec, task, tools, history)
            self.primary_calls += 1
            return action
        except Exception as primary_error:
            try:
                action = self._secondary.next_action(spec, task, tools, history)
                self.secondary_calls += 1
                return action
            except Exception as secondary_error:
                raise secondary_error from primary_error


class CachingBackend:
    """Wraps another backend with a crash-safe, on-disk result cache, so a rerun
    after a crash (or a deliberate resumption) never re-pays for a call it
    already made and recorded.

    The cache key is ``(spec_id, task_id, call_index)`` where ``call_index``
    counts how many times *this process* has already asked for that exact
    (spec, task) pair. That makes it safe for the two shapes this experiment
    needs: a generation's single call per task (call_index is always 0, so a
    rerun after the same generation replays the cached answer instead of
    re-spending) and a noise-floor probe's several independent replicate calls
    on the very same spec (call_index 0, 1, 2, ... are each their own cache
    slot, so replicate 2 is never accidentally served replicate 0's answer).
    Never used to collapse genuinely independent samples into one.
    """

    def __init__(self, backend, cache_path: Path) -> None:
        self._backend = backend
        self._cache_path = cache_path
        self._counts: dict[str, int] = {}
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    @property
    def calls_made(self) -> int:
        return getattr(self._backend, "calls_made", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._backend, "total_tokens", 0)

    def next_action(self, spec, task, tools, history) -> AgentAction:
        pair_key = f"{spec.spec_id}::{task.task_id}"
        index = self._counts.get(pair_key, 0)
        self._counts[pair_key] = index + 1
        cache_key = f"{pair_key}::{index}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return AgentAction(
                final_answer=cached["final_answer"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
            )

        action = self._backend.next_action(spec, task, tools, history)
        self._cache[cache_key] = {
            "final_answer": action.final_answer,
            "prompt_tokens": action.prompt_tokens,
            "completion_tokens": action.completion_tokens,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return action


class SpendCapExceeded(RuntimeError):
    """Raised the instant cumulative spend crosses the configured hard cap.
    Never caught and retried around -- a crossed cap stops the run."""


class MeteringBackend:
    """Wraps another backend to make spend observable and bounded, call by call.

    Every single call is logged (not just once per phase), and cumulative
    token spend is checked after every one: crossing ``max_total_tokens``
    raises :class:`SpendCapExceeded` immediately, mid-run, rather than only
    finding out how much a phase cost after the fact. This exists because a
    prior run was killed without anyone -- including this code -- knowing
    exactly how many of its calls had already landed; this closes that gap.
    """

    def __init__(self, backend, *, max_total_tokens: int, log=print) -> None:
        self._backend = backend
        self._max_total_tokens = max_total_tokens
        self._log = log
        self.calls_logged = 0

    @property
    def calls_made(self) -> int:
        return getattr(self._backend, "calls_made", 0)

    @property
    def total_tokens(self) -> int:
        return getattr(self._backend, "total_tokens", 0)

    def next_action(self, spec, task, tools, history) -> AgentAction:
        action = self._backend.next_action(spec, task, tools, history)
        self.calls_logged += 1
        total = self.total_tokens
        self._log(
            f"    [spend] call {self.calls_logged} ({spec.spec_id}::{task.task_id}): "
            f"+{action.prompt_tokens + action.completion_tokens} tokens, cumulative {total}"
        )
        if total > self._max_total_tokens:
            raise SpendCapExceeded(
                f"cumulative spend {total} tokens exceeded the cap of "
                f"{self._max_total_tokens} after call {self.calls_logged} "
                f"({spec.spec_id}::{task.task_id}); stopping immediately"
            )
        return action

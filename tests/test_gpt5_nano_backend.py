"""Tests for experiments/gpt5_nano_backend.py: the real-model ModelBackend.

None of these make a network call. They exercise the wiring that matters --
missing-key handling, response parsing into an AgentAction, and the retry
path -- with ``requests.post`` mocked out. The backend is unused by default:
nothing in the main test suite or in ``agent_engineer`` itself imports it, and
it only does anything once a caller sets ``OPENAI_API_KEY`` and constructs it
explicitly (see ``experiments/run_real_model_gate.py`` and the README).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from gpt5_nano_backend import ApiKeyMissing, GPT5NanoBackend  # noqa: E402


class _Task:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


class _Spec:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


def _fake_response(*, content: str, prompt_tokens: int = 12, completion_tokens: int = 7) -> Mock:
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return response


def test_missing_key_raises_without_simulating_one(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        GPT5NanoBackend()


def test_next_action_parses_a_real_response_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()

    post = Mock(return_value=_fake_response(content="def f(): return 1"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = backend.next_action(_Spec("system text"), _Task("user text"), (), ())

    assert action.final_answer == "def f(): return 1"
    assert action.prompt_tokens == 12
    assert action.completion_tokens == 7
    assert action.is_final
    assert backend.calls_made == 1

    # the system/user split reaches the API untouched, and reasoning_effort is set
    _, kwargs = post.call_args
    body = kwargs["json"]
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert body["reasoning_effort"] == "minimal"


def test_next_action_retries_transient_failures_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    post = Mock(
        side_effect=[
            ConnectionError("transient"),
            _fake_response(content="ok"),
        ]
    )
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    action = backend.next_action(_Spec("s"), _Task("u"), (), ())

    assert action.final_answer == "ok"
    assert post.call_count == 2


def test_next_action_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    backend = GPT5NanoBackend()
    monkeypatch.setattr("gpt5_nano_backend.time.sleep", lambda _seconds: None)

    post = Mock(side_effect=ConnectionError("still down"))
    monkeypatch.setattr("gpt5_nano_backend.requests.post", post)

    with pytest.raises(ConnectionError):
        backend.next_action(_Spec("s"), _Task("u"), (), ())

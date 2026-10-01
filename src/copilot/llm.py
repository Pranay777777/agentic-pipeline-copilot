"""A minimal client for OpenAI-compatible chat APIs - OpenRouter first.

One endpoint, no SDK: easy to fake and explicit about retries. Free tiers
rate limit hard, so 429 and 5xx responses are retried with backoff that
honours `Retry-After`; authentication and request errors are not, because
retrying them cannot help. Every request caps its output tokens.

The key lives in a `SecretStr` and only in the Authorization header - never
in an exception, a log line or a repr.

`ScriptedLLM` replays fixed responses: the tests use it, and it is the seed
of deterministic replay (step 76).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import SecretStr

RETRYABLE = frozenset({429, 500, 502, 503, 504})
Message = dict[str, str]


class LLMError(RuntimeError):
    """The provider could not produce a response."""


class EmptyCompletionError(LLMError):
    """A 200 with no text - free models do this under load, so it is retried."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> Completion: ...


class OpenRouterLLM:
    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_tokens: int = 3000,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise LLMError(
                "no API key - set OPENROUTER_API_KEY (a free key from openrouter.ai/keys)"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "HTTP-Referer": "https://github.com/Pranay777777/agentic-pipeline-copilot",
                "X-Title": "agentic-pipeline-copilot",
            },
        )

    def complete(self, messages: list[Message]) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if last:
                    raise LLMError(f"could not reach the provider: {type(exc).__name__}") from exc
                self._sleep(float(2**attempt))
                continue
            if response.status_code in RETRYABLE and not last:
                self._sleep(_backoff(response, attempt))
                continue
            if response.status_code >= 400:
                raise LLMError(f"provider returned {response.status_code}: {_reason(response)}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMError("provider returned a body that is not JSON") from exc
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict) and not payload.get("choices"):
                # OpenRouter can report an upstream failure inside a 200 response.
                code = error.get("code")
                status = code if isinstance(code, int) else 502
                if status in RETRYABLE and not last:
                    self._sleep(float(2**attempt))
                    continue
                raise LLMError(f"provider returned {status}: {str(error.get('message'))[:300]}")
            try:
                return _parse(payload, self.model)
            except EmptyCompletionError:
                if last:
                    raise
                self._sleep(float(2**attempt))
        raise LLMError("retries exhausted")  # pragma: no cover - the loop returns or raises


def _backoff(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after", "")
    if retry_after.replace(".", "", 1).isdigit():
        return min(float(retry_after), 60.0)
    return float(2**attempt)


def _reason(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.reason_phrase
    return str(message)[:300]


def _parse(payload: Any, model: str) -> Completion:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        text = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError("unexpected response shape") from exc
    if not isinstance(text, str) or not text.strip():
        finish = choice.get("finish_reason")
        detail = f"finish_reason={finish}"
        if message.get("reasoning"):
            detail += ", it returned reasoning only"
        hint = ""
        if finish == "length":
            hint = " - raise LLM_MAX_TOKENS or choose a model that does not reason at length"
        raise EmptyCompletionError(f"the model returned an empty message ({detail}){hint}")
    usage = payload.get("usage") or {}
    return Completion(
        text=text,
        model=str(payload.get("model") or model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


@dataclass
class ScriptedLLM:
    """Returns the given responses in order and records every request."""

    responses: Sequence[str]
    model: str = "scripted"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: list[Message]) -> Completion:
        if len(self.calls) >= len(self.responses):
            raise LLMError("the script has no more responses")
        self.calls.append([dict(m) for m in messages])
        text = self.responses[len(self.calls) - 1]
        return Completion(text, self.model, prompt_tokens=10, completion_tokens=5)

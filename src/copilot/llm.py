"""A minimal client for OpenAI-compatible chat APIs - OpenRouter first, any compatible one too.

`provider="openrouter"` adds OpenRouter's extras (the `reasoning` object,
`models` fallbacks, attribution headers); `provider="openai"` sends only the
standard fields plus `reasoning_effort`, for endpoints such as Google AI
Studio's OpenAI-compatible Gemini API.

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
        timeout: float = 180.0,
        max_tokens: int = 8000,
        max_retries: int = 3,
        reasoning_effort: str | None = "low",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        fallbacks: Sequence[str] = (),
        provider: str = "openrouter",
    ) -> None:
        if not api_key.get_secret_value():
            raise LLMError(
                "no API key - set OPENROUTER_API_KEY (a free key from openrouter.ai/keys), "
                "or LLM_API_KEY for another provider"
            )
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.fallbacks = [m for m in fallbacks if m and m != model]
        """OpenRouter tries these, in order, when the first model is rate limited or down."""
        """Asks reasoning models to think briefly: free ones otherwise spend the whole
        token cap on hidden reasoning and return no answer. Ignored by other models."""
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"}
            | (
                {
                    "HTTP-Referer": "https://github.com/Pranay777777/agentic-pipeline-copilot",
                    "X-Title": "agentic-pipeline-copilot",
                }
                if provider == "openrouter"
                else {}
            ),
        )

    def complete(self, messages: list[Message]) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "openrouter":
            if self.fallbacks:
                body["models"] = [self.model, *self.fallbacks]
            if self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        elif self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
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
    """Honour Retry-After; otherwise wait longer for rate limits than for outages.

    Free tiers rate limit per minute upstream, so 1-2-4 s retries all land in the
    same window; 5-15-45 s give the window time to pass.
    """
    retry_after = response.headers.get("retry-after", "")
    if retry_after.replace(".", "", 1).isdigit():
        return min(float(retry_after), 60.0)
    if response.status_code == 429:
        return min(5.0 * 3.0**attempt, 60.0)
    return float(2**attempt)


def _reason(response: httpx.Response) -> str:
    """The provider's message, plus the upstream detail OpenRouter nests in metadata."""
    try:
        error = response.json()["error"]
        message = str(error["message"])
    except (ValueError, KeyError, TypeError):
        return response.reason_phrase
    metadata = error.get("metadata") or {}
    raw = " ".join(str(metadata.get("raw", "")).split())
    upstream = metadata.get("provider_name")
    if raw or upstream:
        message += f" [{upstream or 'upstream'}: {raw[:200] or 'no detail'}]"
    return message[:400]


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

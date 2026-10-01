"""Deterministic replay (step 76): record every model call of a run, replay it with no model.

A cassette is a JSONL file, one line per call:

    {"key": "<sha256 of the request>", "prompt": "<last message, abridged>",
     "text": "...", "model": "...", "prompt_tokens": 1200, "completion_tokens": 340}

The key is the SHA-256 of the request's messages (canonical JSON), so replay
answers exactly the request that was recorded: if a prompt, the catalog or a
correction message changes, the key changes and replay **fails loudly**
instead of answering a different question. Identical requests are answered in
recorded order. Replay needs no key and no network, which is what makes a
recorded run a free, hermetic CI test (ADR-007).

    python -m copilot.agents run "..." --record runs/orders.cassette.jsonl
    python -m copilot.agents run "..." --replay runs/orders.cassette.jsonl
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from copilot.llm import LLM, Completion, LLMError, Message

PREVIEW = 160


def request_key(messages: list[Message]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RecordingLLM:
    """Passes each call through and appends it to the cassette (a new file per run)."""

    def __init__(self, inner: LLM, path: Path) -> None:
        self.inner, self.path = inner, path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def complete(self, messages: list[Message]) -> Completion:
        completion = self.inner.complete(messages)
        last = messages[-1]["content"] if messages else ""
        entry = {
            "key": request_key(messages),
            "prompt": " ".join(last.split())[:PREVIEW],
            "text": completion.text,
            "model": completion.model,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as cassette:
            cassette.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return completion


class ReplayLLM:
    """Answers only recorded requests, in recorded order; anything else is an error."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise LLMError(f"no cassette at {path} - record one with --record")
        self.path = path
        self._answers: dict[str, deque[Completion]] = defaultdict(deque)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                completion = Completion(
                    str(entry["text"]),
                    str(entry["model"]),
                    int(entry.get("prompt_tokens", 0)),
                    int(entry.get("completion_tokens", 0)),
                )
                self._answers[str(entry["key"])].append(completion)
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError(f"{path}:{number}: not a cassette line ({exc})") from None

    @property
    def remaining(self) -> int:
        return sum(len(q) for q in self._answers.values())

    def complete(self, messages: list[Message]) -> Completion:
        key = request_key(messages)
        queue = self._answers.get(key)
        if not queue:
            raise LLMError(
                f"{self.path} has no recorded answer for this request (key {key[:12]}) - a prompt, "
                "the catalog or the code changed since it was recorded; record it again"
            )
        return queue.popleft()

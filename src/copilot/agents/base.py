"""What every agent shares: a view of the catalog, and a bounded JSON exchange."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from copilot.catalog.model import CatalogDoc, Kind
from copilot.llm import LLM, Message

T = TypeVar("T", bound=BaseModel)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class Catalog:
    """The catalog as the agents' checks need it: ids by kind, columns by table."""

    def __init__(self, docs: Sequence[CatalogDoc]) -> None:
        self.docs = {d.id: d for d in docs}
        self.columns = {
            d.id.removeprefix("table:"): frozenset(d.tags) for d in docs if d.kind is Kind.TABLE
        }

    def ids(self, kind: Kind) -> list[str]:
        return sorted(i for i, d in self.docs.items() if d.kind is kind)

    def is_a(self, doc_id: str, kind: Kind) -> bool:
        doc = self.docs.get(doc_id)
        return doc is not None and doc.kind is kind


@dataclass(frozen=True)
class Attempt:
    """One model answer and what the deterministic check said about it."""

    stage: str
    number: int
    model: str
    errors: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int


class StageRejectedError(RuntimeError):
    """The stage's check rejected every answer the model gave within its budget."""

    def __init__(self, stage: str, errors: Sequence[str], attempts: Sequence[Attempt]) -> None:
        self.stage = stage
        self.errors = list(errors)
        self.attempts = list(attempts)
        super().__init__(f"{stage} rejected after {len(attempts)} attempt(s): " + "; ".join(errors))


@dataclass
class Exchange:
    """A conversation with one agent, kept so a rejection can be explained."""

    stage: str
    attempts: list[Attempt] = field(default_factory=list)


def extract_json(text: str) -> object:
    """The JSON object in a model answer, with or without a Markdown fence."""
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the answer")
    return json.loads(body[start : end + 1])


def ask_json(
    llm: LLM,
    stage: str,
    messages: list[Message],
    model: type[T],
    check: Callable[[T], list[str]],
    max_attempts: int,
    exchange: Exchange | None = None,
) -> T:
    """Ask until the answer parses as `model` and `check` finds nothing, or give up."""
    exchange = exchange or Exchange(stage)
    conversation = list(messages)
    errors: list[str] = []
    for number in range(1, max_attempts + 1):
        completion = llm.complete(conversation)
        result: T | None = None
        try:
            result = model.model_validate(extract_json(completion.text))
            errors = check(result)
        except ValueError as exc:  # JSONDecodeError and ValidationError are ValueErrors
            errors = _describe(exc)
        exchange.attempts.append(
            Attempt(
                stage,
                number,
                completion.model,
                tuple(errors),
                completion.prompt_tokens,
                completion.completion_tokens,
            )
        )
        if result is not None and not errors:
            return result
        conversation += [
            {"role": "assistant", "content": completion.text},
            {
                "role": "user",
                "content": "That answer was rejected:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nReturn the corrected JSON object only.",
            },
        ]
    raise StageRejectedError(stage, errors, exchange.attempts)


def _describe(exc: ValueError) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}" for e in exc.errors()
        ][:10]
    return [f"not valid JSON: {exc}"]

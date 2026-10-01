"""Cost governor (step 77): a per-run budget of tokens and model calls, with a hard stop.

Every model call of a run goes through `GovernedLLM`. Before a call it
refuses once the run has used its calls or its tokens; after a call it counts
the tokens the provider reported and stops the run at once if they went over
- the answer that crossed the line is not used. A stopped run ends
`rejected` at stage `budget`, with the usage in its report, so a runaway
correction loop or a model that rambles costs at most one call past the
line, never an open-ended bill (ADR-007).

Each completion is already capped by `LLM_MAX_TOKENS`; this caps the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from copilot.llm import LLM, Completion, LLMError, Message


@dataclass(frozen=True)
class Budget:
    max_tokens: int = 60_000
    max_calls: int = 8


class BudgetExceededError(LLMError):
    """The run hit its token or call budget; no further model call is made."""


class GovernedLLM:
    def __init__(self, inner: LLM, budget: Budget) -> None:
        self.inner, self.budget = inner, budget
        self.calls = 0
        self.tokens = 0

    def reset(self) -> None:
        self.calls = self.tokens = 0

    def usage(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "max_calls": self.budget.max_calls,
            "tokens": self.tokens,
            "max_tokens": self.budget.max_tokens,
        }

    def complete(self, messages: list[Message]) -> Completion:
        if self.calls >= self.budget.max_calls:
            raise BudgetExceededError(
                f"call budget spent: {self.calls} of {self.budget.max_calls} model calls"
            )
        if self.tokens >= self.budget.max_tokens:
            raise BudgetExceededError(
                f"token budget spent: {self.tokens} of {self.budget.max_tokens} tokens"
            )
        completion = self.inner.complete(messages)
        self.calls += 1
        self.tokens += completion.prompt_tokens + completion.completion_tokens
        if self.tokens > self.budget.max_tokens:
            raise BudgetExceededError(
                f"token budget exceeded: {self.tokens} of {self.budget.max_tokens} tokens "
                f"after {self.calls} call(s); the last answer was not used"
            )
        return completion

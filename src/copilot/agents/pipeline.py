"""Plan -> generate -> review, with every stage bounded and recorded (ADR-002).

A run ends in one of two states. **ready**: the plan matched the catalog,
the draft's citations held and the Critic found nothing - the notebook goes
on to the sandbox (step 69). **rejected**: a stage could not get past its
check within its budget; the run says which stage and why, and keeps every
attempt, so a human (or the agent eval) can see where it went wrong. Nothing
is retried silently and nothing is delivered half-checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from copilot.agents.base import Attempt, Catalog, Exchange, StageRejectedError
from copilot.agents.critic import Review, review
from copilot.agents.generator import Generator, NotebookDraft, render
from copilot.agents.planner import Plan, Planner
from copilot.catalog.index import CatalogIndex
from copilot.llm import LLM


@dataclass
class Run:
    spec: str
    status: Literal["ready", "rejected"] = "rejected"
    stage: str | None = None
    """The stage that rejected the run, if it was rejected."""
    reasons: list[str] = field(default_factory=list)
    plan: Plan | None = None
    draft: NotebookDraft | None = None
    notebook: str | None = None
    reviews: list[Review] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.spec,
            "status": self.status,
            "stage": self.stage,
            "reasons": self.reasons,
            "model_calls": len(self.attempts),
            "tokens": self.tokens,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "attempts": [
                {"stage": a.stage, "number": a.number, "model": a.model, "errors": list(a.errors)}
                for a in self.attempts
            ],
            "reviews": [[str(f) for f in r.findings] for r in self.reviews],
        }


class Pipeline:
    def __init__(
        self, llm: LLM, catalog: Catalog, max_attempts: int = 2, critic_rounds: int = 1
    ) -> None:
        index = CatalogIndex(list(catalog.docs.values()))
        self.planner = Planner(llm, catalog, index, max_attempts)
        self.generator = Generator(llm, catalog, max_attempts)
        self.critic_rounds = critic_rounds

    def run(self, spec: str) -> Run:
        result = Run(spec)
        exchange = Exchange("run", result.attempts)
        try:
            result.plan = self.planner.plan(spec, exchange)
            result.draft = self.generator.generate(result.plan, exchange)
            for revision in range(self.critic_rounds + 1):
                result.notebook = render(result.draft, result.plan)
                verdict = review(result.notebook)
                result.reviews.append(verdict)
                if verdict.passed:
                    result.status = "ready"
                    return result
                if revision == self.critic_rounds:
                    result.stage, result.reasons = "review", [str(f) for f in verdict.findings]
                    return result
                result.draft = self.generator.generate(
                    result.plan,
                    exchange,
                    review=[str(f) for f in verdict.findings],
                    previous=result.draft,
                )
        except StageRejectedError as exc:
            result.stage, result.reasons = exc.stage, exc.errors
        return result

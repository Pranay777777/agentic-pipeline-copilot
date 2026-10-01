"""Plan -> generate -> review -> execute, with a bounded self-correction loop (ADR-002, ADR-005).

A run ends in one of four states:

- **ready** - the plan matched the catalog, the draft's citations held, the
  Critic found nothing and the notebook ran on sample data in the sandbox
  and met the plan's expectations;
- **reviewed** - as ready, but run without a sandbox, so it never executed;
  not deliverable, useful for trying prompts;
- **escalated** - the notebook was produced, but review findings or
  execution errors were still there after the last correction round; a
  human gets a diagnostic report with the whole trajectory;
- **rejected** - a stage could not produce an artefact its check accepts.

Every model answer, review and execution is kept, so the correction
trajectory is part of the record (step 71), not something reconstructed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from copilot.agents.base import Attempt, Catalog, Exchange, StageRejectedError
from copilot.agents.critic import Review, review
from copilot.agents.generator import Generator, NotebookDraft, render
from copilot.agents.planner import Plan, Planner
from copilot.agents.validator import ValidationReport, Validator
from copilot.catalog.index import CatalogIndex
from copilot.llm import LLM

Status = Literal["ready", "reviewed", "escalated", "rejected"]


@dataclass
class Round:
    """One pass of a notebook through review and, if it passed review, execution."""

    number: int
    review: Review
    validation: ValidationReport | None = None

    @property
    def feedback(self) -> list[str]:
        if not self.review.passed:
            return [str(f) for f in self.review.findings]
        if self.validation is not None:
            return [str(e) for e in self.validation.errors]
        return []


@dataclass
class Run:
    spec: str
    status: Status = "rejected"
    stage: str | None = None
    """The stage that stopped the run: plan, generate, review or validate."""
    reasons: list[str] = field(default_factory=list)
    plan: Plan | None = None
    draft: NotebookDraft | None = None
    notebook: str | None = None
    rounds: list[Round] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)

    @property
    def corrections(self) -> int:
        return max(len(self.rounds) - 1, 0)

    def summary(self) -> dict[str, object]:
        return {
            "spec": self.spec,
            "status": self.status,
            "stage": self.stage,
            "reasons": self.reasons,
            "model_calls": len(self.attempts),
            "tokens": self.tokens,
            "corrections": self.corrections,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "attempts": [
                {"stage": a.stage, "number": a.number, "model": a.model, "errors": list(a.errors)}
                for a in self.attempts
            ],
            "rounds": [
                {
                    "round": r.number,
                    "review": [str(f) for f in r.review.findings],
                    "execution": None
                    if r.validation is None
                    else {
                        "errors": [str(e) for e in r.validation.errors],
                        "checks": list(r.validation.checks),
                        "seconds": r.validation.seconds,
                    },
                }
                for r in self.rounds
            ],
        }


class Pipeline:
    def __init__(
        self,
        llm: LLM,
        catalog: Catalog,
        max_attempts: int = 2,
        max_corrections: int = 2,
        validator: Validator | None = None,
    ) -> None:
        index = CatalogIndex(list(catalog.docs.values()))
        self.planner = Planner(llm, catalog, index, max_attempts)
        self.generator = Generator(llm, catalog, max_attempts)
        self.validator = validator
        self.max_corrections = max_corrections

    def run(self, spec: str, on_attempt: Callable[[Attempt], None] | None = None) -> Run:
        result = Run(spec)
        exchange = Exchange("run", result.attempts, on_attempt)
        try:
            result.plan = self.planner.plan(spec, exchange)
            result.draft = self.generator.generate(result.plan, exchange)
            for number in range(self.max_corrections + 1):
                current = self._round(number, result.plan, result.draft, result)
                if not current.feedback:
                    result.status = "ready" if self.validator else "reviewed"
                    return result
                if number == self.max_corrections:
                    result.status = "escalated"
                    result.stage = "review" if not current.review.passed else "validate"
                    result.reasons = current.feedback
                    return result
                result.draft = self.generator.generate(
                    result.plan, exchange, review=current.feedback, previous=result.draft
                )
        except StageRejectedError as exc:
            result.stage, result.reasons = exc.stage, exc.errors
        return result

    def _round(self, number: int, plan: Plan, draft: NotebookDraft, result: Run) -> Round:
        result.notebook = render(draft, plan)
        current = Round(number, review(result.notebook))
        if current.review.passed and self.validator is not None:
            current.validation = self.validator.validate(plan, result.notebook)
        result.rounds.append(current)
        return current


def diagnostic_report(run: Run) -> str:
    """What a human needs when the loop gives up: the trajectory, then the last notebook."""
    lines = [
        f"# Copilot run: {run.status}",
        "",
        f"**Spec:** {run.spec}",
        "",
        f"**Stopped at:** {run.stage or '-'} after {run.corrections} correction round(s), "
        f"{len(run.attempts)} model call(s), {run.tokens} tokens.",
        "",
        "## Why",
        "",
        *[f"- {r}" for r in run.reasons],
        "",
        "## Trajectory",
        "",
    ]
    for a in run.attempts:
        verdict = "accepted" if not a.errors else "; ".join(a.errors)
        lines.append(f"- {a.stage} #{a.number} ({a.model}): {verdict}")
    for r in run.rounds:
        lines += ["", f"### Round {r.number}", ""]
        lines += [f"- review: {f}" for f in r.review.findings] or ["- review: passed"]
        if r.validation is not None:
            lines += [f"- execution: {e}" for e in r.validation.errors] or ["- execution: passed"]
            lines += [
                f"- check {c['name']}: {'ok' if c['passed'] else 'failed'} ({c['detail']})"
                for c in r.validation.checks
            ]
    if run.plan is not None:
        lines += ["", "## Plan", "", "```json", run.plan.model_dump_json(indent=2), "```"]
    if run.notebook is not None:
        lines += ["", "## Last notebook", "", "```python", run.notebook.rstrip(), "```"]
    return "\n".join(lines) + "\n"

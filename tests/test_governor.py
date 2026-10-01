"""Cost governor: a run stops at its budget of tokens or model calls, and says so."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.__main__ import main
from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline, diagnostic_report
from copilot.governor import Budget, BudgetExceededError, GovernedLLM
from copilot.llm import ScriptedLLM

Make = Callable[..., dict[str, Any]]
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins"
ASK = [{"role": "user", "content": "x"}]


def test_calls_and_tokens_are_counted_and_capped() -> None:
    governed = GovernedLLM(ScriptedLLM(["a", "b", "c"]), Budget(max_tokens=100, max_calls=2))
    governed.complete(ASK)
    governed.complete(ASK)
    assert governed.usage() == {"calls": 2, "max_calls": 2, "tokens": 30, "max_tokens": 100}
    with pytest.raises(BudgetExceededError, match="call budget spent: 2 of 2"):
        governed.complete(ASK)
    governed.reset()
    assert governed.usage()["calls"] == 0
    tight = GovernedLLM(ScriptedLLM(["a", "b"]), Budget(max_tokens=20, max_calls=8))
    tight.complete(ASK)
    with pytest.raises(BudgetExceededError, match=r"30 of 20 tokens after 2 call\(s\)"):
        tight.complete(ASK)
    with pytest.raises(BudgetExceededError, match="token budget spent: 30 of 20"):
        tight.complete(ASK)


def test_a_run_over_budget_is_stopped_with_its_usage(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    result = Pipeline(llm, catalog, budget=Budget(max_calls=1)).run(SPEC)
    assert (result.status, result.stage) == ("rejected", "budget")
    assert result.reasons == ["call budget spent: 1 of 1 model calls"]
    assert result.plan is not None and result.draft is None and len(llm.calls) == 1
    assert result.summary()["budget"] == {
        "calls": 1,
        "max_calls": 1,
        "tokens": 15,
        "max_tokens": 60_000,
    }
    assert "**Budget:** 1 of 1 calls, 15 of 60000 tokens." in diagnostic_report(result)
    within = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    assert Pipeline(within, catalog).run(SPEC).usage["calls"] == 2


def test_cli_budget_comes_from_the_settings(
    monkeypatch: pytest.MonkeyPatch,
    make_plan: Make,
    make_draft: Make,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from copilot.config import get_settings

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("RUN_MAX_TOKENS", "10")
    get_settings.cache_clear()
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    report = tmp_path / "why.md"
    assert main(["run", SPEC, "--no-sandbox", "--report", str(report)], llm=llm) == 1
    assert "rejected at budget" in capsys.readouterr().err
    assert "token budget exceeded: 15 of 10 tokens" in report.read_text("utf-8")
    get_settings.cache_clear()

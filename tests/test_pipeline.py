"""The run end to end with a scripted model and a fake sandbox."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.__main__ import main, make_llm, make_sandbox
from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline, diagnostic_report
from copilot.agents.validator import Validator
from copilot.config import Settings
from copilot.llm import LLMError, OpenRouterLLM, ScriptedLLM
from copilot.sandbox.runner import DockerSandbox

Make = Callable[..., dict[str, Any]]
Results = dict[str, dict[str, Any]]
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins."
REPO = Path(__file__).resolve().parents[1]


def appending(make_draft: Make) -> dict[str, Any]:
    draft = make_draft()
    draft["cells"][2]["code"] = 'deduped.write.mode("append").saveAsTable(target_table)'
    return draft


def script(*answers: dict[str, Any]) -> ScriptedLLM:
    return ScriptedLLM([json.dumps(a) for a in answers])


def test_a_notebook_that_runs_and_meets_expectations_is_ready(
    catalog: Catalog, make_plan: Make, make_draft: Make, sandbox_factory: Any, results: Results
) -> None:
    sandbox = sandbox_factory(results["passed"])
    llm = script(make_plan(), make_draft())
    result = Pipeline(llm, catalog, validator=Validator(sandbox, catalog)).run(SPEC)
    assert (result.status, result.corrections) == ("ready", 0)
    assert result.rounds[0].validation is not None and result.rounds[0].validation.passed
    assert "# COMMAND ----------" in sandbox.notebooks[0]
    assert result.tokens == 30


def test_without_a_sandbox_the_best_a_run_can_be_is_reviewed(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    result = Pipeline(script(make_plan(), make_draft()), catalog).run(SPEC)
    assert result.status == "reviewed" and result.rounds[0].validation is None


def test_execution_errors_feed_the_next_correction(
    catalog: Catalog, make_plan: Make, make_draft: Make, sandbox_factory: Any, results: Results
) -> None:
    sandbox = sandbox_factory(results["cell_error"], results["passed"])
    llm = script(make_plan(), make_draft(), make_draft())
    result = Pipeline(llm, catalog, validator=Validator(sandbox, catalog)).run(SPEC)
    assert (result.status, result.corrections) == ("ready", 1)
    feedback = llm.calls[2][1]["content"]
    assert "cell 2 'Read new Bronze rows' (notebook line 4): AnalysisException" in feedback
    assert "# It was sent back for these reasons" in feedback


def test_review_findings_are_fixed_before_anything_runs(
    catalog: Catalog, make_plan: Make, make_draft: Make, sandbox_factory: Any, results: Results
) -> None:
    sandbox = sandbox_factory(results["passed"])
    llm = script(make_plan(), appending(make_draft), make_draft())
    result = Pipeline(llm, catalog, validator=Validator(sandbox, catalog)).run(SPEC)
    assert result.status == "ready" and len(sandbox.jobs) == 1
    assert result.rounds[0].validation is None and not result.rounds[0].review.passed
    assert 'mode("append") duplicates on re-run' in llm.calls[2][1]["content"]


def test_two_failed_corrections_escalate_with_the_whole_trajectory(
    catalog: Catalog, make_plan: Make, make_draft: Make, sandbox_factory: Any, results: Results
) -> None:
    sandbox = sandbox_factory(results["duplicates"], results["cell_error"], results["duplicates"])
    llm = script(make_plan(), make_draft(), make_draft(), make_draft())
    result = Pipeline(llm, catalog, validator=Validator(sandbox, catalog)).run(SPEC)
    assert (result.status, result.stage, result.corrections) == ("escalated", "validate", 2)
    assert result.reasons == ["expectation: one_row_per_key: 12 order_id value(s) twice"]
    report = diagnostic_report(result)
    assert report.startswith("# Copilot run: escalated")
    assert "after 2 correction round(s), 4 model call(s)" in report
    assert "### Round 1" in report and "execution: cell 2 'Read new Bronze rows'" in report
    assert "check one_row_per_key: failed" in report and "## Last notebook" in report
    summary = result.summary()
    assert summary["corrections"] == 2 and len(summary["rounds"]) == 3  # type: ignore[arg-type]


def test_review_findings_left_after_the_last_round_escalate(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    bad = appending(make_draft)
    result = Pipeline(script(make_plan(), bad, bad, bad), catalog).run(SPEC)
    assert (result.status, result.stage) == ("escalated", "review")
    assert "writes-are-idempotent" in result.reasons[0]


def test_a_plan_the_catalog_rejects_stops_the_run(catalog: Catalog, make_plan: Make) -> None:
    wrong = make_plan(sources=["payments"])
    result = Pipeline(script(wrong, wrong), catalog).run(SPEC)
    assert (result.status, result.stage) == ("rejected", "plan")
    assert result.draft is None and len(result.attempts) == 2
    assert any("'payments' is not a catalog table" in r for r in result.reasons)
    assert "## Last notebook" not in diagnostic_report(result)


# --- CLI ---------------------------------------------------------------


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(REPO)
    return tmp_path


def test_cli_run_writes_a_ready_notebook_and_its_trace(
    repo: Path,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: Results,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out, trace = repo / "nb" / "orders_silver.py", repo / "trace.json"
    argv = ["run", SPEC, "--out", str(out), "--trace", str(trace)]
    llm = script(make_plan(), make_draft())
    assert main(argv, llm=llm, sandbox=sandbox_factory(results["passed"])) == 0
    captured = capsys.readouterr()
    assert (
        "ready:" in captured.out and "(2 model call(s), 30 tokens, 0 correction(s))" in captured.out
    )
    assert "plan #1: ok - scripted, 15 tokens" in captured.err
    assert out.read_text(encoding="utf-8").startswith("# Databricks notebook source")
    assert json.loads(trace.read_text(encoding="utf-8"))["status"] == "ready"
    assert main(["review", str(out)]) == 0


def test_cli_without_sandbox_says_reviewed(
    repo: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    out = repo / "nb.py"
    llm = script(make_plan(), make_draft())
    assert main(["run", SPEC, "--no-sandbox", "--out", str(out)], llm=llm) == 0
    assert capsys.readouterr().out.startswith("reviewed:")


def test_cli_escalation_writes_a_diagnostic_report(
    repo: Path,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: Results,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sandbox = sandbox_factory(*[results["duplicates"]] * 3)
    llm = script(make_plan(), make_draft(), make_draft(), make_draft())
    report = repo / "why.md"
    assert main(["run", SPEC, "--report", str(report)], llm=llm, sandbox=sandbox) == 1
    err = capsys.readouterr().err
    assert "escalated at validate" in err and f"diagnostic report: {report}" in err
    assert "one_row_per_key" in report.read_text(encoding="utf-8")


def test_cli_reports_rejections_and_plans(
    repo: Path, make_plan: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    wrong = make_plan(keys=["nope"])
    assert main(["plan", SPEC], llm=script(wrong, wrong)) == 1
    assert "rejected at plan:" in capsys.readouterr().err
    assert main(["plan", SPEC], llm=script(make_plan())) == 0
    assert '"target": "orders_silver"' in capsys.readouterr().out


def test_cli_review_flags_a_bad_notebook(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    notebook = repo / "bad.py"
    notebook.write_text('rows = df.collect()\npassword = "x"\n', encoding="utf-8")
    assert main(["review", str(notebook)]) == 1
    out = capsys.readouterr().out
    assert f"{notebook}:line 1: collect()" in out and "2 finding(s)" in out


def test_cli_needs_docker_a_key_and_reports_provider_errors(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from copilot.config import get_settings

    missing = DockerSandbox("no-such-image:0", docker="no-such-docker-binary")
    assert main(["run", SPEC], llm=ScriptedLLM([]), sandbox=missing) == 2
    assert "python -m copilot.sandbox build" in capsys.readouterr().err
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()
    assert main(["run", SPEC, "--no-sandbox"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert main(["run", SPEC, "--no-sandbox"], llm=ScriptedLLM([])) == 2
    assert "no more responses" in capsys.readouterr().err
    get_settings.cache_clear()


def test_factories_use_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "some/model:free")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", "b:free, c:free")
    monkeypatch.setenv("SANDBOX_TIMEOUT_S", "99")
    settings = Settings()
    llm = make_llm(settings)
    assert isinstance(llm, OpenRouterLLM) and llm.model == "some/model:free"
    assert llm.fallbacks == ["b:free", "c:free"]
    sandbox = make_sandbox(settings)
    assert sandbox.image == "copilot-sandbox:0.2" and sandbox.limits.timeout_s == 99
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "g-key")
    gemini = make_llm(Settings())
    assert isinstance(gemini, OpenRouterLLM) and gemini.provider == "openai"
    with pytest.raises(LLMError):
        make_llm(Settings(openrouter_api_key="", llm_api_key="", _env_file=None))  # type: ignore[call-arg,arg-type]

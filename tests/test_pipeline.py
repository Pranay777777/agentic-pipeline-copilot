"""The run end to end with a scripted model: ready, revised, or rejected with a reason."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.__main__ import main, make_llm
from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline
from copilot.config import Settings
from copilot.llm import LLMError, OpenRouterLLM, ScriptedLLM

Make = Callable[..., dict[str, Any]]
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins."
REPO = Path(__file__).resolve().parents[1]


def appending(make_draft: Make) -> dict[str, Any]:
    draft = make_draft()
    draft["cells"][2]["code"] = 'deduped.write.mode("append").saveAsTable(silver_table)'
    return draft


def test_a_clean_run_is_ready(catalog: Catalog, make_plan: Make, make_draft: Make) -> None:
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    result = Pipeline(llm, catalog).run(SPEC)
    assert result.status == "ready" and result.stage is None
    assert result.notebook is not None and "# COMMAND ----------" in result.notebook
    assert [a.stage for a in result.attempts] == ["plan", "generate"]
    assert result.tokens == 30 and len(result.reviews) == 1


def test_a_critic_finding_goes_back_to_the_generator_once(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    llm = ScriptedLLM(
        [json.dumps(make_plan()), json.dumps(appending(make_draft)), json.dumps(make_draft())]
    )
    result = Pipeline(llm, catalog, critic_rounds=1).run(SPEC)
    assert result.status == "ready" and len(result.reviews) == 2
    assert not result.reviews[0].passed and result.reviews[1].passed
    assert "blind append" not in llm.calls[2][1]["content"]
    assert 'mode("append") duplicates on re-run' in llm.calls[2][1]["content"]


def test_findings_left_after_the_last_round_reject_the_run(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    bad = json.dumps(appending(make_draft))
    llm = ScriptedLLM([json.dumps(make_plan()), bad, bad])
    result = Pipeline(llm, catalog, critic_rounds=1).run(SPEC)
    assert (result.status, result.stage) == ("rejected", "review")
    assert result.reasons and "writes-are-idempotent" in result.reasons[0]
    summary = result.summary()
    assert summary["model_calls"] == 3 and len(summary["reviews"]) == 2  # type: ignore[arg-type]


def test_a_plan_the_catalog_rejects_stops_the_run(catalog: Catalog, make_plan: Make) -> None:
    wrong = json.dumps(make_plan(sources=["payments"]))
    result = Pipeline(ScriptedLLM([wrong, wrong]), catalog).run(SPEC)
    assert (result.status, result.stage) == ("rejected", "plan")
    assert result.draft is None and len(result.attempts) == 2
    assert any("'payments' is not a catalog table" in r for r in result.reasons)


# --- CLI ---------------------------------------------------------------


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(REPO)
    return tmp_path


def test_cli_run_writes_a_ready_notebook_and_its_trace(
    repo: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    out, trace = repo / "nb" / "orders_silver.py", repo / "trace.json"
    argv = ["run", SPEC, "--out", str(out), "--trace", str(trace)]
    assert main(argv, llm=llm) == 0
    captured = capsys.readouterr()
    assert "ready:" in captured.out and "(2 model call(s), 30 tokens)" in captured.out
    assert "plan #1: ok - scripted, 15 tokens" in captured.err
    assert "generate #1: ok" in captured.err
    assert out.read_text(encoding="utf-8").startswith("# Databricks notebook source")
    assert json.loads(trace.read_text(encoding="utf-8"))["status"] == "ready"
    assert main(["review", str(out)]) == 0


def test_cli_reports_rejections(
    repo: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = json.dumps(appending(make_draft))
    llm = ScriptedLLM([json.dumps(make_plan()), bad, bad])
    assert main(["run", SPEC], llm=llm) == 1
    err = capsys.readouterr().err
    assert "rejected at review (3 model call(s)" in err and "generate #1: ok" in err
    wrong = json.dumps(make_plan(keys=["nope"]))
    assert main(["plan", SPEC], llm=ScriptedLLM([wrong, wrong])) == 1
    assert "rejected at plan:" in capsys.readouterr().err
    assert main(["plan", SPEC], llm=ScriptedLLM([json.dumps(make_plan())])) == 0
    assert '"target": "orders_silver"' in capsys.readouterr().out


def test_cli_review_flags_a_bad_notebook(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    notebook = repo / "bad.py"
    notebook.write_text('rows = df.collect()\npassword = "x"\n', encoding="utf-8")
    assert main(["review", str(notebook)]) == 1
    out = capsys.readouterr().out
    assert f"{notebook}:line 1: collect()" in out and "2 finding(s)" in out


def test_cli_needs_a_key_and_reports_provider_errors(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from copilot.config import get_settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()
    assert main(["run", SPEC]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert main(["run", SPEC], llm=ScriptedLLM([])) == 2
    assert "no more responses" in capsys.readouterr().err
    get_settings.cache_clear()


def test_make_llm_uses_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "some/model:free")
    llm = make_llm(Settings())
    assert isinstance(llm, OpenRouterLLM) and llm.model == "some/model:free"
    with pytest.raises(LLMError):
        make_llm(Settings(openrouter_api_key="", _env_file=None))  # type: ignore[call-arg,arg-type]

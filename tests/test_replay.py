"""Deterministic replay: a recorded run answers again with no model, and only the same requests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.__main__ import main
from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline
from copilot.agents.validator import Validator
from copilot.llm import LLMError, ScriptedLLM
from copilot.replay import RecordingLLM, ReplayLLM, request_key

Make = Callable[..., dict[str, Any]]
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins"
LIVE = Path(__file__).resolve().parent / "cassettes" / "orders_silver.jsonl"
"""A real run, recorded with --record; committed so CI replays it for free."""


def scripted(make_plan: Make, make_draft: Make) -> ScriptedLLM:
    return ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])


def test_a_recorded_run_replays_identically_with_no_model(
    catalog: Catalog,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: dict[str, Any],
    tmp_path: Path,
) -> None:
    tape = tmp_path / "run.jsonl"
    live = Pipeline(
        RecordingLLM(scripted(make_plan, make_draft), tape),
        catalog,
        validator=Validator(sandbox_factory(results["passed"]), catalog),
    ).run(SPEC)
    entries = [json.loads(line) for line in tape.read_text("utf-8").splitlines()]
    assert len(entries) == 2 and entries[0]["prompt"] and entries[0]["prompt_tokens"] == 10
    replay = ReplayLLM(tape)
    again = Pipeline(
        replay, catalog, validator=Validator(sandbox_factory(results["passed"]), catalog)
    ).run(SPEC)
    assert again.status == live.status == "ready" and again.notebook == live.notebook
    assert again.tokens == live.tokens and replay.remaining == 0


def test_only_the_recorded_request_gets_an_answer(tmp_path: Path) -> None:
    tape = tmp_path / "tape.jsonl"
    recorder = RecordingLLM(ScriptedLLM(["first", "second"]), tape)
    ask = [{"role": "user", "content": "plan it"}]
    recorder.complete(ask)
    recorder.complete(ask)
    replay = ReplayLLM(tape)
    assert [replay.complete(ask).text, replay.complete(ask).text] == ["first", "second"]
    with pytest.raises(LLMError, match=r"no recorded answer .* record it again"):
        replay.complete([{"role": "user", "content": "plan it differently"}])
    assert request_key(ask) == request_key([{"content": "plan it", "role": "user"}])


def test_missing_and_broken_cassettes_are_errors(tmp_path: Path) -> None:
    with pytest.raises(LLMError, match="no cassette at"):
        ReplayLLM(tmp_path / "none.jsonl")
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"key": "k", "text": "t", "model": "m"}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(LLMError, match=r"broken.jsonl:3: not a cassette line"):
        ReplayLLM(broken)


def test_cli_records_then_replays_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from copilot.config import get_settings

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    tape, out = tmp_path / "orders.jsonl", tmp_path / "nb" / "orders_silver.py"
    argv = ["run", SPEC, "--out", str(out)]
    llm = scripted(make_plan, make_draft)
    sandbox = sandbox_factory(results["passed"])
    assert main([*argv, "--record", str(tape)], llm=llm, sandbox=sandbox) == 0
    recorded = out.read_text("utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    capsys.readouterr()
    sandbox = sandbox_factory(results["passed"])
    assert main([*argv, "--replay", str(tape)], sandbox=sandbox) == 0
    captured = capsys.readouterr()
    assert f"replaying {tape} - no model, no network" in captured.err
    assert captured.out.startswith("ready:") and out.read_text("utf-8") == recorded
    assert main([*argv, "--replay", str(tmp_path / "none.jsonl")], sandbox=sandbox) == 2
    assert "no cassette at" in capsys.readouterr().err
    get_settings.cache_clear()


@pytest.mark.skipif(not LIVE.exists(), reason="no live cassette recorded yet")
def test_the_committed_live_run_still_replays(catalog: Catalog) -> None:  # pragma: no cover
    """Fails when a prompt or the catalog changes - re-record with --record when it does."""
    replay = ReplayLLM(LIVE)
    result = Pipeline(replay, catalog).run(SPEC)
    assert result.status == "reviewed", result.reasons
    assert result.plan is not None and result.notebook is not None

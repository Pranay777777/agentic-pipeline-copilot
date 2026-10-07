"""Eval suite: record in batches, replay for free, gate on regressions and the pass rate."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from copilot.evals import load_specs, main
from copilot.llm import LLM, LLMError, ScriptedLLM

Make = Callable[..., dict[str, Any]]
REPO = Path(__file__).resolve().parents[1]
EXPECT = {
    "strategy": "dedup_latest",
    "layer": "silver",
    "sources": ["orders"],
    "keys": ["order_id"],
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "catalog").mkdir()
    shutil.copy(REPO / "catalog" / "snapshot.jsonl", tmp_path / "catalog" / "snapshot.jsonl")
    specs = [
        {"id": "a", "spec": "Silver orders, newest per order_id", "expect": EXPECT},
        {"id": "b", "spec": "Dedup orders into Silver on order_id", "expect": EXPECT},
        {"id": "c", "spec": "Orders again", "expect": EXPECT},
    ]
    (tmp_path / "evals").mkdir()
    write_specs(tmp_path, specs)
    return tmp_path


def write_specs(root: Path, specs: list[dict[str, Any]]) -> None:
    (root / "evals" / "specs.json").write_text(json.dumps({"specs": specs}), encoding="utf-8")


def models(*scripts: list[str]) -> Callable[[], LLM]:
    queue: Iterator[list[str]] = iter(scripts)
    return lambda: ScriptedLLM(next(queue))


def test_the_committed_suite_has_fifty_unique_specs_with_expectations() -> None:
    specs = load_specs(REPO / "evals" / "specs.json")
    assert len(specs) == 50
    allowed = {"strategy", "layer", "sources", "keys"}
    assert all(s["expect"] and set(s["expect"]) <= allowed for s in specs)


def test_record_in_batches_then_replay_and_gate(
    root: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    assert main(["record", "--limit", "2"], make=models(good, good), root=root) == 0
    assert "recorded 2; 1 of 3 specs still unrecorded" in capsys.readouterr().out
    assert main(["record", "--limit", "5"], make=models(good), root=root) == 0
    assert "recorded 1; 0 of 3" in capsys.readouterr().out

    assert main(["run"], root=root) == 0
    report = json.loads((root / "evals" / "report.json").read_text("utf-8"))
    assert report["summary"]["passed"] == 3 and report["summary"]["first_try_rate"] == 1.0
    assert report["outcomes"][0]["calls"] == 2 and report["outcomes"][0]["tokens"] == 30

    assert main(["gate", "--update-baseline"], root=root) == 0
    assert "baseline: 3 passing spec(s)" in capsys.readouterr().out
    assert main(["gate"], root=root) == 0
    assert capsys.readouterr().out.endswith("gate: pass\n")
    assert main(["render"], root=root) == 0
    svg = (root / "docs" / "images" / "eval-report.svg").read_text("utf-8")
    assert svg.startswith("<svg") and "suite" in svg and "first&#160;try" in svg


def test_a_regression_or_a_low_rate_fails_the_gate(
    root: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    main(["record"], make=models(good, good, good), root=root)
    main(["gate", "--update-baseline"], root=root)
    specs = load_specs(root / "evals" / "specs.json")
    specs[0]["expect"] = {**EXPECT, "strategy": "scd2"}
    write_specs(root, specs)
    capsys.readouterr()
    assert main(["gate"], root=root) == 1
    captured = capsys.readouterr()
    assert "regression: a passed in the baseline, now fails - strategy: expected scd2" in (
        captured.err
    )
    assert "gate: FAIL" in captured.out
    assert main(["gate", "--threshold", "0.9"], root=root) == 1
    assert "pass rate 67% of 3 recorded is below 90%" in capsys.readouterr().err


def test_a_changed_spec_makes_its_cassette_stale(
    root: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    main(["record", "--only", "a"], make=models(good), root=root)
    specs = load_specs(root / "evals" / "specs.json")
    specs[0]["spec"] = "Silver orders, reworded"
    write_specs(root, specs)
    capsys.readouterr()
    assert main(["run"], root=root) == 0
    assert "FAIL a: " in capsys.readouterr().err and "record it again" in (
        (root / "evals" / "report.json").read_text("utf-8")
    )
    assert main(["record", "--only", "zzz"], root=root) == 2


def test_recording_stops_at_the_first_provider_error(
    root: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    assert main(["record"], make=models(good, [json.dumps(make_plan())]), root=root) == 2
    captured = capsys.readouterr()
    assert "recorded 1; 2 of 3" in captured.out and "stopped: b: " in captured.err
    assert not (root / "evals" / "cassettes" / "b.jsonl").exists()


def test_a_failure_and_its_recovery_is_rendered(
    root: Path, make_plan: Make, make_draft: Make, capsys: pytest.CaptureFixture[str]
) -> None:
    recovering = [
        json.dumps(make_plan(target="orders")),
        json.dumps(make_plan()),
        json.dumps(make_draft()),
    ]
    rejected = [json.dumps(make_plan(target="orders"))] * 2
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    main(["record"], make=models(recovering, rejected, good), root=root)
    main(["run"], root=root)
    report = json.loads((root / "evals" / "report.json").read_text("utf-8"))
    a, b, _ = report["outcomes"]
    assert a["passed"] and not a["first_try"] and a["attempts"][0]["ok"] is False
    assert not b["passed"] and b["mismatches"][0].startswith("status rejected at plan:")
    main(["render"], root=root)
    svg = (root / "docs" / "images" / "recovery.svg").read_text("utf-8")
    assert "would" in svg and "overwrite" in svg


class FailingLLM:
    """A provider that fails the way the real client reports it."""

    def __init__(self, error: LLMError) -> None:
        self.error = error
        self.model = "failing"

    def complete(self, messages: Any) -> Any:
        raise self.error


def providers(*llms: LLM) -> Callable[[], LLM]:
    queue: Iterator[LLM] = iter(llms)
    return lambda: next(queue)


UNAVAILABLE = LLMError("provider returned 503: high demand", status=503, transient=True)
TIMEOUT = LLMError("could not reach the provider: ReadTimeout", transient=True)
SCHEMA = LLMError("provider returned 400: response_format is invalid", status=400)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("copilot.evals.time.sleep", slept.append)
    return slept


def test_a_503_then_success_records(
    root: Path,
    make_plan: Make,
    make_draft: Make,
    waits: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    make = providers(FailingLLM(UNAVAILABLE), ScriptedLLM(good))
    assert main(["record", "--only", "a"], make=make, root=root) == 0
    captured = capsys.readouterr()
    assert "recorded 1;" in captured.out and "pending" not in captured.out
    assert "a: attempt 1/3 - provider returned 503" in captured.err
    assert waits == [5.0]
    assert (root / "evals" / "cassettes" / "a.jsonl").exists()


def test_three_timeouts_leave_a_spec_pending_and_the_next_one_records(
    root: Path,
    make_plan: Make,
    make_draft: Make,
    waits: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    make = providers(*[FailingLLM(TIMEOUT)] * 3, ScriptedLLM(good))
    assert main(["record", "--limit", "1"], make=make, root=root) == 0
    captured = capsys.readouterr()
    assert "recorded 1; 2 of 3 specs still unrecorded" in captured.out
    assert "pending (transient): a" in captured.out
    assert waits == [5.0, 15.0, 45.0]
    cassettes = root / "evals" / "cassettes"
    assert not (cassettes / "a.jsonl").exists() and (cassettes / "b.jsonl").exists()
    # a pending spec is unrecorded, never a failure
    assert main(["gate"], root=root) == 0
    report = json.loads((root / "evals" / "report.json").read_text("utf-8"))
    assert report["summary"]["recorded"] == 1 and report["summary"]["passed"] == 1


def test_a_429_counts_as_transient(
    root: Path, make_plan: Make, make_draft: Make, waits: list[float]
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    limited = LLMError("provider returned 429: slow down", status=429, transient=True)
    make = providers(FailingLLM(limited), FailingLLM(limited), ScriptedLLM(good))
    assert main(["record", "--only", "a"], make=make, root=root) == 0
    assert waits == [5.0, 15.0]


def test_a_schema_error_still_stops_the_batch(
    root: Path,
    make_plan: Make,
    make_draft: Make,
    waits: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = [json.dumps(make_plan()), json.dumps(make_draft())]
    make = providers(ScriptedLLM(good), FailingLLM(SCHEMA), ScriptedLLM(good))
    assert main(["record"], make=make, root=root) == 2
    captured = capsys.readouterr()
    assert (
        "recorded 1; 2 of 3" in captured.out and "stopped: b: provider returned 400" in captured.err
    )
    assert waits == []  # not retried
    assert not (root / "evals" / "cassettes" / "c.jsonl").exists()  # the batch stopped at b

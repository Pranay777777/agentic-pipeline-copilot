"""Eval suite: record in batches, replay for free, gate on regressions and the pass rate."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from copilot.evals import load_specs, main
from copilot.llm import LLM, ScriptedLLM

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

"""Sandbox: the docker flags, the time limit, the harness and the isolation check."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.generator import NotebookDraft, render
from copilot.agents.planner import Plan
from copilot.sandbox import harness
from copilot.sandbox.__main__ import isolation
from copilot.sandbox.__main__ import main as sandbox_main
from copilot.sandbox.runner import DockerSandbox, Execution, Limits, docker_command

Make = Callable[..., dict[str, Any]]


def test_the_container_is_started_with_every_isolation_flag(tmp_path: Path) -> None:
    command = docker_command("img:1", tmp_path / "job", tmp_path / "out", Limits(), "box")
    flags = " ".join(command)
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 1000:1000",
        "--cpus 2.0",
        "--memory 2g",
        "--memory-swap 2g",
        "--pids-limit 512",
        "--tmpfs /tmp:rw,exec,nosuid,nodev,size=1g",
        "--rm",
        "--hostname localhost",
        "--env SPARK_LOCAL_IP=127.0.0.1",
    ):
        assert flag in flags, flag
    assert f"{(tmp_path / 'job').resolve()}:/job:ro" in command
    assert f"{(tmp_path / 'out').resolve()}:/out:rw" in command
    assert command[-3:] == ["img:1", "python", "/job/harness.py"]
    assert not any(c.startswith("--env-file") for c in command)
    assert [c for c in command if c.startswith("HOME=")] == ["HOME=/tmp"]
    envs = [command[i + 1] for i, c in enumerate(command) if c == "--env"]
    assert all(not any(w in e.upper() for w in ("KEY", "TOKEN", "SECRET")) for e in envs)


class Recorder:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes, self.calls = list(outcomes), []  # type: ignore[var-annotated]

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        self.calls.append((command, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def done(code: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, err)


def test_a_run_reports_exit_code_output_and_time(tmp_path: Path) -> None:
    ticks = iter([10.0, 13.5])
    runner = Recorder(done(0, "ok", ""))
    sandbox = DockerSandbox("img:1", runner=runner, clock=lambda: next(ticks))
    (tmp_path / "out").mkdir()
    result = sandbox.run(tmp_path / "job", tmp_path / "out")
    assert result == Execution(0, "ok", "", False, 3.5)
    command, kwargs = runner.calls[0]
    assert command[:2] == ["docker", "run"] and kwargs["timeout"] == 240.0


def test_the_time_limit_kills_the_container(tmp_path: Path) -> None:
    runner = Recorder(subprocess.TimeoutExpired("docker", 5, output=b"partial"), done(0))
    sandbox = DockerSandbox("img:1", Limits(timeout_s=5), runner=runner)
    (tmp_path / "out").mkdir()
    result = sandbox.run(tmp_path / "job", tmp_path / "out")
    assert result.timed_out and result.exit_code == -1 and result.stdout == "partial"
    name = runner.calls[0][0][runner.calls[0][0].index("--name") + 1]
    assert runner.calls[1][0] == ["docker", "kill", name]


def test_missing_docker_and_images_are_reported(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    absent = DockerSandbox("img:1", runner=Recorder(FileNotFoundError(), FileNotFoundError()))
    assert "install Docker" in absent.run(tmp_path / "job", tmp_path / "out").stderr
    assert not absent.available()
    assert DockerSandbox("img:1", runner=Recorder(done(0))).available()
    assert not DockerSandbox("img:1", runner=Recorder(done(1))).available()


# --- harness (the parts that do not need Spark) ---------------------------------


def test_cells_are_split_from_the_rendered_notebook(make_plan: Make, make_draft: Make) -> None:
    text = render(NotebookDraft.model_validate(make_draft()), Plan.model_validate(make_plan()))
    cells = harness.split_cells(text)
    assert [t for t, _, _ in cells] == [
        "Parameters",
        "Read new Bronze rows",
        "Newest row per order",
        "MERGE into Silver",
    ]
    lines = text.splitlines()
    for title, code, first in cells:
        assert lines[first - 1] == f"# DBTITLE 1,{title}"
        assert "\n".join(lines[first - 1 : first - 1 + len(code.splitlines())]) == code
    assert harness.split_cells("x = 1")[0] == ("cell 1", "x = 1", 1)


def test_cells_run_in_one_namespace_and_stop_at_the_first_error() -> None:
    ns: dict[str, Any] = {}
    cells = [("a", "x = 1", 3), ("b", "y = x + 1\n\nz = y / 0", 10), ("c", "never = True", 20)]
    outcome = harness.run_cells(cells, ns)
    assert outcome["ran"] == ["a"] and ns["y"] == 2 and "never" not in ns
    assert outcome["error"] == {
        "stage": "cell",
        "cell": 2,
        "title": "b",
        "line": 12,
        "kind": "ZeroDivisionError",
        "message": "division by zero",
    }
    assert harness.run_cells([("a", "x = 1", 1)], {})["error"] is None
    plan_noise = ValueError("[UNRESOLVED_COLUMN] `x` cannot be resolved;\n'Project [a#1]\n+- x")
    assert harness.first_line(plan_noise) == "[UNRESOLVED_COLUMN] `x` cannot be resolved"


def test_dbutils_stand_in_gives_widgets_and_refuses_secrets() -> None:
    dbutils = harness.DBUtils({"target_table": "orders_silver"})
    dbutils.widgets.text("target_table", "")
    dbutils.widgets.text("extra", "fallback")
    assert dbutils.widgets.get("target_table") == "orders_silver"
    assert dbutils.widgets.get("extra") == "fallback"
    with pytest.raises(KeyError, match="never defined"):
        dbutils.widgets.get("nope")
    with pytest.raises(PermissionError, match="not available in the sandbox"):
        dbutils.secrets.get("kv", "db-password")
    assert harness.ddl([["id", "string"], ["n", "int64"], ["odd", "elapsed"]]) == (
        "`id` STRING, `n` BIGINT, `odd` STRING"
    )


# --- sandbox CLI -----------------------------------------------------------------


class ProbeSandbox:
    def __init__(self, probe: dict[str, Any] | None) -> None:
        self.probe = probe

    def run(self, job: Path, out: Path) -> Execution:
        assert json.loads((job / "job.json").read_text("utf-8")) == {"mode": "probe"}
        if self.probe is not None:
            (out / "result.json").write_text(json.dumps({"probe": self.probe}), "utf-8")
        return Execution(0, "", "daemon not running", False, 1.0)


ISOLATED = {
    "network": "blocked (OSError)",
    "dns": "blocked (gaierror)",
    "write_job": "blocked (OSError)",
    "write_root": "blocked (OSError)",
    "write_out": "allowed",
    "write_tmp": "allowed",
    "uid": 1000,
    "env_secrets": [],
}


def test_check_passes_only_when_every_escape_is_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    assert sandbox_main(["check"], sandbox=ProbeSandbox(ISOLATED)) == 0
    assert capsys.readouterr().out.strip().endswith("isolated")
    for leak in ({"network": "allowed"}, {"uid": 0}, {"env_secrets": ["OPENROUTER_API_KEY"]}):
        ok, _ = isolation(ProbeSandbox(ISOLATED | leak))
        assert not ok, leak
    assert sandbox_main(["check"], sandbox=ProbeSandbox(None)) == 1
    assert "daemon not running" in capsys.readouterr().out


def test_build_runs_docker_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return done(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert sandbox_main(["build"]) == 0
    assert seen == [["docker", "build", "--tag", "copilot-sandbox:0.1", "sandbox"]]
    assert "docker build" in capsys.readouterr().out

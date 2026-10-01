"""Runs a notebook inside the sandbox container - and only there (ADR-005).

Copied into each job directory and started as `python /job/harness.py` in a
container with no network, a read-only root filesystem and hard limits. It
must not import copilot: the container has only PySpark and Delta.

It reads /job/job.json, runs the static-analysis gate (ruff and mypy) on the
notebook, and only if that is clean loads the sample tables, runs the
notebook's cells in order in one namespace (with `spark` and a `dbutils`
stand-in), runs the generated pytest file against the target table, and
writes /out/result.json (ADR-006).
Executing code with `exec` is acceptable here precisely because this process
is the isolation boundary's inside; the host never executes generated code.

Pure helpers (cell splitting, cell execution, dbutils, the static gate and the
in-process pytest run) are unit-tested on the host; the Spark parts run only
in the container.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Protocol

JOB = Path("/job")
OUT = Path("/out")
SEPARATOR = "# COMMAND ----------"
TITLE = "# DBTITLE 1,"
PUBLIC_ENV = frozenset({"GPG_KEY"})
"""Set by the official python base image: the public fingerprint of the key that signs
Python releases. Named like a secret, but published - any other match fails the check."""
STATIC = Path("/tmp/static")  # noqa: S108 - the container's only writable scratch space
BUILTINS = ("spark", "dbutils", "display")
PRELUDE = (
    "from typing import Any\n"
    "from pyspark.sql import SparkSession\n"
    "spark: SparkSession\n"
    "dbutils: Any\n"
    "def display(*args: Any, **kwargs: Any) -> None: ...\n"
)
"""Declares what Databricks injects, so mypy checks the notebook as Databricks runs it."""
RUFF = (
    "check",
    "--no-cache",
    "--isolated",
    "--select",
    "F,E9,B,S",
    "--ignore",
    "S101,S608",
    "--config",
    f"builtins={list(BUILTINS)!r}",
    "--output-format",
    "json",
)
MYPY = (
    "--no-incremental",
    f"--cache-dir={os.devnull}",  # /dev/null in the container, nul on a Windows host
    "--ignore-missing-imports",
    "--check-untyped-defs",
    "--no-error-summary",
    "--show-error-codes",
)
MYPY_LINE = re.compile(r"^.+?:(\d+): error: (.*?)  \[([a-z-]+)\]$")
SPARK_TYPES = {
    "string": "STRING",
    "int64": "BIGINT",
    "int32": "INT",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
}


def split_cells(source: str) -> list[tuple[str, str, int]]:
    """(title, code, first line in the file) for every code cell of a Databricks notebook."""
    cells: list[tuple[str, str, int]] = []
    chunk: list[str] = []
    start = 1

    def flush(lines: list[str], first: int) -> None:
        if lines and lines[0].startswith("# Databricks notebook source"):
            lines, first = lines[1:], first + 1
        while lines and not lines[0].strip():
            lines, first = lines[1:], first + 1
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        if not lines or all(line.startswith("# MAGIC") for line in lines if line.strip()):
            return
        title = f"cell {len(cells) + 1}"
        if lines[0].startswith(TITLE):
            title = lines[0][len(TITLE) :].strip()
        cells.append((title, "\n".join(lines), first))

    for number, line in enumerate(source.splitlines(), 1):
        if line.strip() == SEPARATOR:
            flush(chunk, start)
            chunk, start = [], number + 1
        else:
            chunk.append(line)
    flush(chunk, start)
    return cells


def first_line(exc: BaseException) -> str:
    """The exception's message without the query plan Spark appends."""
    text = str(exc).strip().split("\n", 1)[0].rstrip(";")
    return " ".join(text.split())[:600]


def run_cells(cells: list[tuple[str, str, int]], namespace: dict[str, Any]) -> dict[str, Any]:
    """Run cells in order; stop at the first failure and say where, in notebook lines."""
    done: list[str] = []
    for n, (title, code, first) in enumerate(cells, 1):
        filename = f"<cell {n}>"
        try:
            exec(compile(code, filename, "exec"), namespace)  # noqa: S102 - inside the sandbox
        except Exception as exc:
            line = None
            for frame in traceback.extract_tb(exc.__traceback__):
                if frame.filename == filename and frame.lineno is not None:
                    line = first + frame.lineno - 1
            return {
                "ran": done,
                "error": {
                    "stage": "cell",
                    "cell": n,
                    "title": title,
                    "line": line,
                    "kind": type(exc).__name__,
                    "message": first_line(exc),
                },
            }
        done.append(title)
    return {"ran": done, "error": None}


class Widgets:
    """dbutils.widgets with the values the Validator chose."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def text(self, name: str, default: str = "", label: str | None = None) -> None:
        self.values.setdefault(name, default)

    def get(self, name: str) -> str:
        if name not in self.values:
            raise KeyError(f"widget '{name}' was never defined")
        return self.values[name]


class Secrets:
    def get(self, scope: str, key: str) -> str:
        raise PermissionError("secrets are not available in the sandbox")


class DBUtils:
    def __init__(self, values: dict[str, str]) -> None:
        self.widgets = Widgets(values)
        self.secrets = Secrets()


class Run(Protocol):
    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]: ...


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False, timeout=180)  # noqa: S603


def static_check(notebook: Path, workdir: Path = STATIC, run: Run = _run) -> list[dict[str, Any]]:
    """ruff and mypy findings for the notebook, in notebook lines; empty means clean.

    A tool that cannot run is a finding too: the gate fails closed.
    """
    findings: list[dict[str, Any]] = []
    ruff = run([sys.executable, "-m", "ruff", *RUFF, str(notebook)])
    if ruff.returncode in (0, 1):
        for item in json.loads(ruff.stdout or "[]"):
            location = item.get("location") or {}
            findings.append(
                {
                    "tool": "ruff",
                    "code": item.get("code") or "error",
                    "line": location.get("row"),
                    "message": " ".join(str(item.get("message", "")).split())[:600],
                }
            )
    else:
        findings.append(_broken("ruff", ruff))
    workdir.mkdir(parents=True, exist_ok=True)
    checked = workdir / "notebook.py"
    checked.write_text(PRELUDE + notebook.read_text(encoding="utf-8"), encoding="utf-8")
    offset = PRELUDE.count("\n")
    mypy = run([sys.executable, "-m", "mypy", *MYPY, str(checked)])
    if mypy.returncode in (0, 1):
        for text in mypy.stdout.splitlines():
            match = MYPY_LINE.match(text.strip())
            if match:
                line = int(match.group(1)) - offset
                findings.append(
                    {
                        "tool": "mypy",
                        "code": match.group(3),
                        "line": line if line > 0 else None,
                        "message": match.group(2)[:600],
                    }
                )
    else:
        findings.append(_broken("mypy", mypy))
    return findings


def _broken(tool: str, done: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    tail = " ".join((done.stderr or done.stdout or "").split())[-600:]
    message = tail or f"exit {done.returncode}"
    return {"tool": tool, "code": "crashed", "line": None, "message": message}


def probe() -> dict[str, Any]:  # pragma: no cover - container only
    """What this process can reach - the evidence that the sandbox is isolated."""

    def attempt(action: Any) -> str:
        try:
            action()
        except Exception as exc:
            return f"blocked ({type(exc).__name__})"
        return "allowed"

    def connect() -> None:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()

    return {
        "network": attempt(connect),
        "dns": attempt(lambda: socket.gethostbyname("pypi.org")),
        "write_job": attempt(lambda: (JOB / "planted").write_text("x")),
        "write_root": attempt(lambda: Path("/etc/planted").write_text("x")),
        "write_out": attempt(lambda: (OUT / "probe").write_text("x")),
        "write_tmp": attempt(lambda: Path("/tmp/probe").write_text("x")),  # noqa: S108
        "uid": getattr(os, "getuid", lambda: -1)(),  # Linux in the container; typed on Windows
        "env_secrets": sorted(
            k
            for k in os.environ
            if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET")) and k not in PUBLIC_ENV
        ),
    }


def spark_session() -> Any:  # pragma: no cover - container only
    # Imported by name: PySpark exists only in the container image.
    spark_sql = importlib.import_module("pyspark.sql")
    jars = ",".join(sorted(str(p) for p in Path("/opt/sandbox/jars").glob("*.jar")))
    return (
        spark_sql.SparkSession.builder.master("local[2]")
        .appName("copilot-sandbox")
        .config("spark.jars", jars)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", "/tmp/warehouse")  # noqa: S108
        .config("spark.local.dir", "/tmp/spark")  # noqa: S108
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def ddl(columns: list[list[str]]) -> str:
    return ", ".join(f"`{name}` {SPARK_TYPES.get(kind, 'STRING')}" for name, kind in columns)


def load_tables(spark: Any, job: dict[str, Any]) -> None:  # pragma: no cover - container only
    for table in job["tables"]:
        rows = json.loads((JOB / "data" / f"{table['source']}.json").read_text(encoding="utf-8"))
        frame = spark.createDataFrame(rows, ddl(table["columns"]))
        frame.write.format("delta").saveAsTable(table["name"])
    target = job["target"]
    spark.createDataFrame([], ddl(job["tables"][0]["columns"])).write.format("delta").saveAsTable(
        target
    )


def run_tests(path: Path, fixtures: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the generated pytest file in this process, against the live session.

    Each test becomes {name, passed, detail}; a skipped test (a contract column
    the target does not carry) counts as passed, a warning is kept in the
    detail. pytest is imported by name: it is in the image, not a harness import.
    """
    pytest = importlib.import_module("pytest")
    outcomes: dict[str, dict[str, Any]] = {}
    warned: dict[str, str] = {}

    def name(nodeid: str) -> str:
        return nodeid.rsplit("::", 1)[-1]

    def spark() -> Any:
        return fixtures["spark"]

    def target_table() -> str:
        return str(fixtures["target_table"])

    def warning_recorded(warning_message: Any, nodeid: str) -> None:
        if nodeid:
            warned[name(nodeid)] = str(warning_message.message)

    def runtest_logreport(report: Any) -> None:
        test = name(report.nodeid)
        if report.when != "call" and not (report.failed or report.skipped):
            return
        if report.failed:
            crash = getattr(getattr(report.longrepr, "reprcrash", None), "message", "")
            lines = (crash or report.longreprtext or "failed").strip().splitlines()
            detail = lines[0] if crash else lines[-1]
        elif report.skipped:
            reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else ""
            detail = f"skipped: {str(reason).removeprefix('Skipped: ')}"
        else:
            detail = "passed"
        outcomes.setdefault(
            test, {"name": test, "passed": not report.failed, "detail": detail[:600]}
        )

    # A module, like a conftest - a class would bind the fixtures as methods. Fixtures
    # are assigned, not decorated: pytest is imported by name, so untyped here.
    plugin = types.ModuleType("copilot_sandbox_fixtures")
    hooks = {
        "spark": pytest.fixture(spark, name="spark"),
        "target_table": pytest.fixture(target_table, name="target_table"),
        "pytest_warning_recorded": warning_recorded,
        "pytest_runtest_logreport": runtest_logreport,
    }
    for attribute, value in hooks.items():
        setattr(plugin, attribute, value)

    sys.modules.pop(path.stem, None)  # a fresh import each run, never a cached module
    args = ["-q", "-p", "no:cacheprovider", "--capture=sys", str(path)]
    code = int(pytest.main(args, plugins=[plugin]))
    if code not in (0, 1) or not outcomes:
        raise RuntimeError(f"pytest exited with code {code} running {path.name}")
    for test, message in warned.items():
        if test in outcomes and outcomes[test]["passed"]:
            outcomes[test]["detail"] = f"warning: {message}"[:600]
    return list(outcomes.values())


def main() -> None:  # pragma: no cover - container only
    job = json.loads((JOB / "job.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {"ran": [], "error": None, "checks": []}
    if job.get("mode") == "probe":
        result["probe"] = probe()
    else:
        started = time.monotonic()
        result["static"] = static_check(JOB / "notebook.py")
        if result["static"]:
            result["seconds"] = round(time.monotonic() - started, 1)
            (OUT / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return
        try:
            spark = spark_session()
            load_tables(spark, job)
        except Exception as exc:
            message = first_line(exc)
            result["error"] = {"stage": "setup", "kind": type(exc).__name__, "message": message}
        else:
            namespace: dict[str, Any] = {"spark": spark, "dbutils": DBUtils(dict(job["params"]))}
            source = (JOB / "notebook.py").read_text(encoding="utf-8")
            result.update(run_cells(split_cells(source), namespace))
            if result["error"] is None:
                try:
                    fixtures = {"spark": spark, "target_table": job["target"]}
                    result["checks"] = run_tests(JOB / "test_notebook.py", fixtures)
                except Exception as exc:
                    message = " ".join(str(exc).split())[:600]
                    result["error"] = {
                        "stage": "check",
                        "kind": type(exc).__name__,
                        "message": message,
                    }
        result["seconds"] = round(time.monotonic() - started, 1)
    (OUT / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()

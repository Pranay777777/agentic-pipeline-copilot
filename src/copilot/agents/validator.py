"""Validator (steps 70, 72): gate, then run the notebook on sample data in the sandbox.

The fourth typed boundary of ADR-002, and the only one that consults
reality. It writes a job directory - the notebook, the in-container harness,
sample rows for every source, the widget values and the plan's expectations
- hands it to the sandbox, and turns what comes back into errors the
Generator can act on: which ruff or mypy finding on which notebook line (the
static gate runs first, in the container - ADR-006), which cell, which
line, which exception, or which
expectation the output broke (one row per key, keys not null, a non-empty
target).

Widget values follow the parameter contract (ADR-005): `<source>_table` is
that source's Bronze table, `target_table` is the plan's target,
`watermark` starts at 0 and `key_columns` lists the plan's keys.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from copilot.agents.base import Catalog
from copilot.agents.planner import Plan, Strategy
from copilot.sandbox import harness
from copilot.sandbox.runner import Execution, Sandbox
from copilot.sandbox.sample import columns, rows

UNIQUE_KEYS = {Strategy.DEDUP_LATEST, Strategy.CDC_MERGE, Strategy.INCREMENTAL, Strategy.SCD2}


def parameters(plan: Plan) -> dict[str, str]:
    """The widget values a notebook receives (the parameter contract)."""
    values = {f"{s}_table": f"bronze_{s}" for s in plan.sources}
    values |= {"target_table": plan.target, "watermark": "0", "key_columns": ",".join(plan.keys)}
    return values


@dataclass(frozen=True)
class ExecutionError:
    stage: str
    """static, setup, cell, check, expectation or sandbox."""
    kind: str
    message: str
    cell: int | None = None
    title: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        where = self.stage
        if self.cell is not None:
            where = f"cell {self.cell} '{self.title}'" + (
                f" (notebook line {self.line})" if self.line else ""
            )
        elif self.line:
            where = f"{self.stage} (notebook line {self.line})"
        return f"{where}: {self.kind}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[ExecutionError, ...]
    checks: tuple[dict[str, Any], ...] = ()
    static: tuple[dict[str, Any], ...] = ()
    """ruff/mypy findings; any of them means the notebook never reached Spark."""
    ran: tuple[str, ...] = ()
    seconds: float = 0.0
    timed_out: bool = False
    log_tail: str = field(default="", repr=False)

    @property
    def passed(self) -> bool:
        return not self.errors


class Validator:
    def __init__(self, sandbox: Sandbox, catalog: Catalog, sample_rows: int = 24) -> None:
        self.sandbox, self.catalog, self.sample_rows = sandbox, catalog, sample_rows

    def job(self, plan: Plan, notebook: str, directory: Path) -> None:
        """Write everything the harness needs into `directory`."""
        (directory / "data").mkdir(parents=True)
        tables = []
        for source in plan.sources:
            cols = columns(self.catalog.docs[f"table:{source}"].text)
            data = rows(cols, plan.keys, self.sample_rows)
            (directory / "data" / f"{source}.json").write_text(json.dumps(data), encoding="utf-8")
            tables.append({"source": source, "name": f"bronze_{source}", "columns": cols})
        spec = {
            "tables": tables,
            "target": plan.target,
            "keys": plan.keys,
            "unique_keys": plan.strategy in UNIQUE_KEYS,
            "params": parameters(plan),
        }
        (directory / "job.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        (directory / "notebook.py").write_text(notebook, encoding="utf-8")
        shutil.copy(Path(harness.__file__), directory / "harness.py")

    def validate(self, plan: Plan, notebook: str) -> ValidationReport:
        with tempfile.TemporaryDirectory(prefix="copilot-") as tmp:
            job, out = Path(tmp) / "job", Path(tmp) / "out"
            out.mkdir()
            self.job(plan, notebook, job)
            execution = self.sandbox.run(job, out)
            result_file = out / "result.json"
            result = json.loads(result_file.read_text("utf-8")) if result_file.exists() else None
        return report(execution, result)


def report(execution: Execution, result: dict[str, Any] | None) -> ValidationReport:
    """Turn the sandbox's exit and the harness's result into a report."""
    tail = (execution.stderr or execution.stdout)[-2000:]
    if execution.timed_out:
        error = ExecutionError("sandbox", "Timeout", f"killed after {execution.seconds:.0f}s")
        return ValidationReport((error,), seconds=execution.seconds, timed_out=True, log_tail=tail)
    if result is None:
        last = " ".join(tail.strip().splitlines()[-3:])[:600] or f"exit code {execution.exit_code}"
        error = ExecutionError("sandbox", "NoResult", last)
        return ValidationReport((error,), seconds=execution.seconds, log_tail=tail)
    static = tuple(result.get("static") or [])
    errors = [
        ExecutionError("static", f"{s['tool']} {s['code']}", s["message"], line=s.get("line"))
        for s in static
    ]
    if result.get("error"):
        e = result["error"]
        errors.append(
            ExecutionError(
                e["stage"], e["kind"], e["message"], e.get("cell"), e.get("title"), e.get("line")
            )
        )
    checks = tuple(result.get("checks") or [])
    errors += [
        ExecutionError("expectation", c["name"], c["detail"]) for c in checks if not c["passed"]
    ]
    return ValidationReport(
        tuple(errors),
        checks,
        static,
        tuple(result.get("ran") or []),
        float(result.get("seconds") or execution.seconds),
        log_tail=tail,
    )

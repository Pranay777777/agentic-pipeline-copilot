"""Validator: sample data from the catalog, the job it writes, and how results become errors."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from copilot.agents.base import Catalog
from copilot.agents.planner import Plan
from copilot.agents.validator import Validator, parameters, report
from copilot.sandbox import harness
from copilot.sandbox.runner import DockerSandbox, Execution
from copilot.sandbox.sample import columns, rows

Make = Callable[..., dict[str, Any]]


# --- sample data ---------------------------------------------------------------


def test_sample_rows_exercise_dedup_joins_and_nulls(catalog: Catalog) -> None:
    cols = columns(catalog.docs["table:orders"].text)
    assert cols[0] == ("order_id", "string") and ("updated_at", "int64") in cols
    data = rows(cols, ["order_id"], 24)
    ids = [r["order_id"] for r in data]
    assert len(data) == 24 and len(set(ids)) == 12 and ids[0] == ids[12] == "order-000"
    assert data[12]["updated_at"] > data[0]["updated_at"]
    assert data[3]["customer_id"] == "customer-003" and data[13]["customer_id"] == "customer-003"
    assert data[6]["order_status"] is None and data[6]["order_id"] is not None
    mixed = rows([("n", "int32"), ("x", "float64"), ("ok", "bool"), ("k", "int64")], ["k"], 4)
    assert mixed[1] == {"n": 1, "x": 11.5, "ok": False, "k": 1}


# --- validator -------------------------------------------------------------------


def test_the_job_holds_notebook_harness_data_and_expectations(
    catalog: Catalog, make_plan: Make, tmp_path: Path
) -> None:
    plan = Plan.model_validate(make_plan())
    Validator(DockerSandbox(), catalog, sample_rows=8).job(plan, "x = 1\n", tmp_path / "job")
    job = json.loads((tmp_path / "job" / "job.json").read_text("utf-8"))
    assert job["target"] == "orders_silver" and job["keys"] == ["order_id"]
    assert job["unique_keys"] is True and job["tables"][0]["name"] == "bronze_orders"
    assert (
        job["params"]
        == parameters(plan)
        == {
            "orders_table": "bronze_orders",
            "target_table": "orders_silver",
            "watermark": "0",
            "key_columns": "order_id",
        }
    )
    data = json.loads((tmp_path / "job" / "data" / "orders.json").read_text("utf-8"))
    assert len(data) == 8
    harness_copy = (tmp_path / "job" / "harness.py").read_text("utf-8")
    assert harness_copy == Path(harness.__file__).read_text("utf-8")


def test_timeouts_and_missing_results_become_errors() -> None:
    slow = report(Execution(-1, "", "", True, 240.0), None)
    assert not slow.passed and str(slow.errors[0]) == "sandbox: Timeout: killed after 240s"
    crashed = report(Execution(125, "", "docker: image not found\nmore detail", False, 1.0), None)
    assert str(crashed.errors[0]) == "sandbox: NoResult: docker: image not found more detail"
    silent = report(Execution(1, "", "", False, 1.0), None)
    assert str(silent.errors[0]) == "sandbox: NoResult: exit code 1"
    setup = report(
        Execution(0, "", "", False, 5.0),
        {"error": {"stage": "setup", "kind": "Py4JError", "message": "no JVM"}, "checks": []},
    )
    assert str(setup.errors[0]) == "setup: Py4JError: no JVM"


def test_static_findings_become_errors_with_their_notebook_line() -> None:
    finding = {"tool": "ruff", "code": "F821", "line": 14, "message": "Undefined name `x`"}
    gated = report(
        Execution(0, "", "", False, 3.0),
        {"ran": [], "error": None, "checks": [], "static": [finding]},
    )
    assert not gated.passed and gated.static == (finding,) and gated.ran == ()
    assert str(gated.errors[0]) == "static (notebook line 14): ruff F821: Undefined name `x`"
    crashed = {"tool": "mypy", "code": "crashed", "line": None, "message": "exit 2"}
    assert str(report(Execution(0, "", "", False, 1.0), {"static": [crashed]}).errors[0]) == (
        "static: mypy crashed: exit 2"
    )

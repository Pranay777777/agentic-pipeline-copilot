"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.base import Catalog
from copilot.catalog.model import read_snapshot
from copilot.config import Settings

SNAPSHOT = Path(__file__).resolve().parents[1] / "catalog" / "snapshot.jsonl"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings isolated from the developer's own .env.

    Environment variables take precedence over the .env file in
    pydantic-settings, so setting them here makes the test deterministic
    regardless of what is on the machine.
    """
    monkeypatch.setenv("APP_ENV", "ci")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    return Settings()


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    """The committed lakehouse catalog the agents are checked against."""
    return Catalog(read_snapshot(SNAPSHOT))


def plan_dict(**changes: Any) -> dict[str, Any]:
    """A valid plan: deduplicate orders into Silver, newest row per order."""
    plan: dict[str, Any] = {
        "intent": "Deduplicate orders into Silver, newest row per order",
        "layer": "silver",
        "strategy": "dedup_latest",
        "sources": ["orders"],
        "target": "orders_silver",
        "keys": ["order_id"],
        "incremental_column": "updated_at",
        "steps": [
            {
                "action": "Read new Bronze orders since the watermark",
                "table": "orders",
                "columns": ["order_id", "customer_id", "order_status", "updated_at"],
                "pattern": "pattern:module.ingest-incremental",
            },
            {
                "action": "Keep the newest row per order_id",
                "table": "orders",
                "columns": ["order_id", "updated_at"],
                "pattern": "pattern:module.transform-silver",
            },
            {"action": "MERGE into Silver on order_id", "pattern": "pattern:module.ingest-cdc"},
        ],
        "standards": [
            "standard:notebook.silver-keeps-one-current-row-per-key",
            "standard:notebook.writes-are-idempotent",
        ],
    }
    plan.update(changes)
    return plan


@pytest.fixture
def make_plan() -> Callable[..., dict[str, Any]]:
    return plan_dict


READ = """from pyspark.sql import functions as F

columns = ["order_id", "customer_id", "order_status", "purchased_at", "delivered_at", "updated_at"]
new_rows = (
    spark.table(orders_table).where(F.col("updated_at") > F.lit(int(watermark))).select(*columns)
)"""
DEDUP = """from pyspark.sql import Window

latest = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc())
ranked = new_rows.withColumn("_rank", F.row_number().over(latest))
deduped = ranked.where("_rank = 1").drop("_rank")"""
MERGE = """from delta.tables import DeltaTable

(
    DeltaTable.forName(spark, target_table)
    .alias("t")
    .merge(deduped.alias("s"), "t.order_id = s.order_id")
    .whenMatchedUpdateAll(condition="s.updated_at > t.updated_at")
    .whenNotMatchedInsertAll()
    .execute()
)"""


def draft_dict(**changes: Any) -> dict[str, Any]:
    """A draft that implements plan_dict() and follows the standards."""
    draft: dict[str, Any] = {
        "parameters": ["orders_table", "target_table", "watermark"],
        "cells": [
            {
                "title": "Read new Bronze rows",
                "code": READ,
                "cites": ["pattern:module.ingest-incremental", "table:orders"],
            },
            {
                "title": "Newest row per order",
                "code": DEDUP,
                "cites": [
                    "pattern:module.transform-silver",
                    "standard:notebook.silver-keeps-one-current-row-per-key",
                ],
            },
            {
                "title": "MERGE into Silver",
                "code": MERGE,
                "cites": ["pattern:module.ingest-cdc", "standard:notebook.writes-are-idempotent"],
            },
        ],
    }
    draft.update(changes)
    return draft


@pytest.fixture
def make_draft() -> Callable[..., dict[str, Any]]:
    return draft_dict


class FakeSandbox:
    """Plays the container: records each job and writes a scripted result."""

    def __init__(self, *results: dict[str, Any] | None, timed_out: bool = False) -> None:
        from copilot.sandbox.runner import Execution

        self.results = list(results)
        self.jobs: list[dict[str, Any]] = []
        self.notebooks: list[str] = []
        self.timed_out = timed_out
        self._execution = Execution

    def run(self, job: Path, out: Path) -> Any:
        import json

        self.jobs.append(json.loads((job / "job.json").read_text("utf-8")))
        notebook = job / "notebook.py"
        self.notebooks.append(notebook.read_text("utf-8") if notebook.exists() else "")
        result = self.results.pop(0) if self.results else None
        if result is not None:
            (out / "result.json").write_text(json.dumps(result), "utf-8")
        return self._execution(0 if result else 1, "", "boom: no result", self.timed_out, 12.0)


PASSED = {
    "ran": ["Parameters", "Read new Bronze rows", "Newest row per order", "MERGE into Silver"],
    "error": None,
    "checks": [
        {"name": "target_not_empty", "passed": True, "detail": "12 rows"},
        {"name": "one_row_per_key", "passed": True, "detail": "0 order_id value(s) ..."},
        {"name": "keys_not_null", "passed": True, "detail": "0 row(s) with a null key"},
    ],
    "seconds": 21.5,
}
CELL_ERROR = {
    "ran": ["Parameters"],
    "error": {
        "stage": "cell",
        "cell": 2,
        "title": "Read new Bronze rows",
        "line": 4,
        "kind": "AnalysisException",
        "message": "[UNRESOLVED_COLUMN] `order_ts` cannot be resolved",
    },
    "checks": [],
    "seconds": 9.0,
}
DUPLICATES = {
    "ran": ["Parameters", "Read new Bronze rows", "Newest row per order", "MERGE into Silver"],
    "error": None,
    "checks": [
        {"name": "target_not_empty", "passed": True, "detail": "24 rows"},
        {"name": "one_row_per_key", "passed": False, "detail": "12 order_id value(s) twice"},
    ],
    "seconds": 20.0,
}


@pytest.fixture
def results() -> dict[str, dict[str, Any]]:
    return {"passed": PASSED, "cell_error": CELL_ERROR, "duplicates": DUPLICATES}


@pytest.fixture
def sandbox_factory() -> type[FakeSandbox]:
    return FakeSandbox


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "docker: needs Docker and the sandbox image (set COPILOT_SANDBOX_IMAGE)"
    )

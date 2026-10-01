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
    spark.table(bronze_table).where(F.col("updated_at") > F.lit(int(watermark))).select(*columns)
)"""
DEDUP = """from pyspark.sql import Window

latest = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc())
ranked = new_rows.withColumn("_rank", F.row_number().over(latest))
deduped = ranked.where("_rank = 1").drop("_rank")"""
MERGE = """from delta.tables import DeltaTable

(
    DeltaTable.forName(spark, silver_table)
    .alias("t")
    .merge(deduped.alias("s"), "t.order_id = s.order_id")
    .whenMatchedUpdateAll(condition="s.updated_at > t.updated_at")
    .whenNotMatchedInsertAll()
    .execute()
)"""


def draft_dict(**changes: Any) -> dict[str, Any]:
    """A draft that implements plan_dict() and follows the standards."""
    draft: dict[str, Any] = {
        "parameters": ["bronze_table", "silver_table", "watermark"],
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

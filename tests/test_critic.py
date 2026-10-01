"""Critic: each rule tied to a catalog standard, reported at its notebook line."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from copilot.agents.base import Catalog
from copilot.agents.critic import (
    COLLECT,
    CONFIG,
    IDEMPOTENT,
    SCHEMA,
    SECRETS,
    review,
)
from copilot.agents.generator import NotebookDraft, render
from copilot.agents.planner import Plan
from copilot.catalog.model import Kind

Make = Callable[..., dict[str, Any]]


def rules(source: str) -> list[tuple[str, int]]:
    return [(f.rule, f.line) for f in review(source).findings]


def test_every_rule_enforces_a_real_catalog_standard(catalog: Catalog) -> None:
    for standard in (COLLECT, SCHEMA, IDEMPOTENT, CONFIG, SECRETS):
        assert catalog.is_a(standard, Kind.STANDARD), standard


def test_a_notebook_that_follows_the_standards_passes(make_plan: Make, make_draft: Make) -> None:
    text = render(NotebookDraft.model_validate(make_draft()), Plan.model_validate(make_plan()))
    assert review(text).passed


@pytest.mark.parametrize(
    ("code", "found"),
    [
        ("rows = df.collect()", [("unbounded-collect", 1)]),
        ("pdf = df.toPandas()", [("unbounded-collect", 1)]),
        ("df.take(n)", [("unbounded-collect", 1)]),
        ("df.head()", [("unbounded-collect", 1)]),
        ("df.take(5000)", [("unbounded-collect", 1)]),
        ("df.take(True)", [("unbounded-collect", 1)]),
        ('df.select("*")', [("select-star", 1)]),
        ('spark.sql("SELECT *\\n FROM t")', [("select-star", 1)]),
        ('df.write.mode("append").saveAsTable(t)', [("blind-append", 1)]),
        ('df.write.saveAsTable(t, mode="append")', [("blind-append", 1)]),
        ('spark.read.parquet("abfss://raw@acct.dfs.core.windows.net/o")', [("hardcoded-path", 1)]),
        ('df.write.save("/mnt/silver/orders")', [("hardcoded-path", 1)]),
        ('password = "hunter2"', [("credential", 1)]),
        ('conn_str: str = "Server=x"', [("credential", 1)]),
        ('connect(api_key="abc123")', [("credential", 1)]),
        ('url = "DefaultEndpointsProtocol=https;AccountKey=abc"', [("credential", 1)]),
        ('cfg.token = "abc"', [("credential", 1)]),
        ("x = 1\ndef f(:\n    pass\n", [("syntax", 2)]),
    ],
)
def test_violations_are_found_at_their_line(code: str, found: list[tuple[str, int]]) -> None:
    assert rules(code) == found


@pytest.mark.parametrize(
    "code",
    [
        "rows = df.limit(20).collect()",
        "first = df.take(10)",
        "sample = df.head(n=100)",
        'df.select("order_id", "updated_at")',
        'df.write.mode("overwrite").saveAsTable(t)',
        'password = dbutils.secrets.get("kv", "db-password")',
        'token = ""',
        'note = "the /mnt/ prefix is not used here"',
        'first, second = "a", "b"',
    ],
)
def test_bounded_and_resolved_code_is_not_flagged(code: str) -> None:
    assert review(code).passed, review(code).findings


def test_findings_name_line_message_and_standard() -> None:
    verdict = review('x = 1\ndf.write.mode("append").save(t)\n')
    assert not verdict.passed
    assert str(verdict.findings[0]) == (
        'line 2: mode("append") duplicates on re-run (standard:notebook.writes-are-idempotent)'
    )

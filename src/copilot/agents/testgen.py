"""Test generation (step 73): every notebook ships with pytest tests, derived by code.

No model writes them (ADR-006). They come from two sources a program can
trust:

- **the plan** - the target is not empty, its keys exist and are never null,
  and incremental, CDC, dedup and SCD2 targets keep one (current) row per key;
- **the catalog's data contracts** - each source column's `Expect:` rules
  (`unique`, `not_null`) become a test on the target, failing or only
  warning according to the rule's `severity`.

The file needs two fixtures, `spark` and `target_table`: the sandbox harness
provides them when it runs the notebook, and a project's conftest provides
them in CI. Because the tests are code-derived they are the same every run,
and a reviewer can read them as the notebook's acceptance criteria.
"""

from __future__ import annotations

import re

from copilot.agents.base import Catalog
from copilot.agents.planner import Plan, Strategy
from copilot.sandbox.sample import contract

UNIQUE_KEYS = {Strategy.DEDUP_LATEST, Strategy.CDC_MERGE, Strategy.INCREMENTAL, Strategy.SCD2}

HEADER = '''"""Tests for `{target}`, generated from its plan and the catalog's data contracts.

Written by agentic-pipeline-copilot (no model involved).
Plan: {strategy} into {layer} from {sources}; keys {keys}.
Fixtures `spark` and `target_table` come from the runner (the sandbox harness,
or the project's conftest). Edit the plan, not this file.
"""

from __future__ import annotations

{imports}from typing import Any

import pytest

KEYS = {keys_list!r}


def current(spark: Any, target_table: str) -> Any:
    """The target's rows - only current ones for SCD2 history tables."""
    frame = spark.table(target_table)
    return frame.where("is_current") if "is_current" in frame.columns else frame


def test_target_is_not_empty(spark: Any, target_table: str) -> None:
    rows = current(spark, target_table).count()
    assert rows > 0, "the notebook wrote no rows"


def test_keys_are_present(spark: Any, target_table: str) -> None:
    missing = [k for k in KEYS if k not in spark.table(target_table).columns]
    assert not missing, f"key column(s) missing from the target: {{missing}}"


def test_keys_are_not_null(spark: Any, target_table: str) -> None:
    condition = " OR ".join(f"`{{k}}` IS NULL" for k in KEYS)
    nulls = current(spark, target_table).where(condition).count()
    assert nulls == 0, f"{{nulls}} row(s) with a null key"
'''

ONE_ROW_PER_KEY = """

def test_one_row_per_key(spark: Any, target_table: str) -> None:
    dupes = current(spark, target_table).groupBy(*KEYS).count().where("count > 1").count()
    assert dupes == 0, f"{dupes} {'/'.join(KEYS)} value(s) appear more than once"
"""

CONTRACT = '''

def test_contract_{name}(spark: Any, target_table: str) -> None:
    """{source}.{column}: {rule} (severity {severity}) - from the data contract."""
    frame = current(spark, target_table)
    if "{column}" not in frame.columns:
        pytest.skip("{column} is not in the target")
    bad = {query}
    if bad:
        message = f"{{bad}} row(s) break {rule} on {column}"
        {action}
'''

QUERIES = {
    "unique": 'frame.groupBy("{column}").count().where("count > 1").count()',
    "not_null": 'frame.where("`{column}` IS NULL").count()',
}


def generate_tests(plan: Plan, catalog: Catalog) -> str:
    """The pytest file that ships with the notebook for `plan`."""
    text = HEADER.replace("{imports}", "{{imports}}").format(
        target=plan.target,
        strategy=plan.strategy,
        layer=plan.layer,
        sources=", ".join(plan.sources),
        keys=", ".join(plan.keys),
        keys_list=list(plan.keys),
    )
    if plan.strategy in UNIQUE_KEYS:
        text += ONE_ROW_PER_KEY
    seen: set[str] = set()
    for source in plan.sources:
        for column, rules in contract(catalog.docs[f"table:{source}"].text).items():
            severity = str(rules.get("severity", "fail"))
            for rule in QUERIES:
                name = re.sub(r"[^a-z0-9_]", "_", f"{column}_{rule}")
                if not rules.get(rule) or name in seen:
                    continue
                seen.add(name)
                action = (
                    "warnings.warn(message, stacklevel=1)"
                    if severity == "warn"
                    else "pytest.fail(message)"
                )
                text += CONTRACT.format(
                    name=name,
                    source=source,
                    column=column,
                    rule=rule,
                    severity=severity,
                    query=QUERIES[rule].format(column=column),
                    action=action,
                )
    imports = "import warnings\n" if "warnings.warn(" in text else ""
    return text.replace("{imports}", imports, 1)

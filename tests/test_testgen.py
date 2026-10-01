"""Generated tests: derived from the plan and the catalog's data contracts, run like the sandbox."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from copilot.agents.base import Catalog
from copilot.agents.planner import Plan
from copilot.agents.testgen import generate_tests
from copilot.sandbox import harness
from copilot.sandbox.sample import columns, contract

Make = Callable[..., dict[str, Any]]


def customers_plan(make_plan: Make, **changes: Any) -> Plan:
    fields = {
        "sources": ["customers"],
        "target": "customers_silver",
        "keys": ["customer_id"],
        **changes,
    }
    return Plan.model_validate(make_plan(**fields))


def names(text: str) -> list[str]:
    tree = ast.parse(text)
    return [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
    ]


# --- the catalog's column lines and contracts --------------------------------------


def test_described_columns_and_their_contracts_are_read(catalog: Catalog) -> None:
    text = catalog.docs["table:customers"].text
    cols = columns(text)
    assert cols[0] == ("customer_id", "string") and cols[-1] == ("updated_at", "int64")
    assert len(cols) == 9
    assert contract(text) == {"customer_id": {"unique": True, "severity": "fail"}}
    assert contract(catalog.docs["table:orders"].text) == {}


# --- what gets generated -------------------------------------------------------------


def test_a_dedup_plan_gets_the_plan_tests_only(catalog: Catalog, make_plan: Make) -> None:
    text = generate_tests(Plan.model_validate(make_plan()), catalog)
    assert names(text) == [
        "test_target_is_not_empty",
        "test_keys_are_present",
        "test_keys_are_not_null",
        "test_one_row_per_key",
    ]
    assert "KEYS = ['order_id']" in text and "import warnings" not in text
    assert text.startswith('"""Tests for `orders_silver`')


def test_a_full_reload_keeps_no_one_row_per_key_rule(catalog: Catalog, make_plan: Make) -> None:
    plan = Plan.model_validate(make_plan(strategy="full_reload", incremental_column=None))
    assert "test_one_row_per_key" not in names(generate_tests(plan, catalog))


def test_contract_rules_become_tests_that_fail(catalog: Catalog, make_plan: Make) -> None:
    text = generate_tests(customers_plan(make_plan), catalog)
    assert names(text)[-1] == "test_contract_customer_id_unique"
    assert "pytest.fail(message)" in text and "import warnings" not in text
    assert "customers.customer_id: unique (severity fail)" in text


def test_warn_severity_only_warns(catalog: Catalog, make_plan: Make) -> None:
    doc = catalog.docs["table:customers"]
    warned = doc.model_copy(
        update={
            "text": doc.text.replace("'severity': 'fail'", "'severity': 'warn', 'not_null': True")
        }
    )
    text = generate_tests(customers_plan(make_plan), Catalog([*catalog.docs.values(), warned]))
    assert names(text)[-2:] == [
        "test_contract_customer_id_unique",
        "test_contract_customer_id_not_null",
    ]
    assert text.count("warnings.warn(message, stacklevel=1)") == 2 and "import warnings\n" in text


# --- the generated file, run the way the harness runs it -----------------------------


class Frame:
    """Just enough of a Spark DataFrame for the generated tests."""

    def __init__(self, rows: list[dict[str, Any]], cols: list[str] | None = None) -> None:
        self.rows = rows
        self.columns = cols if cols is not None else list(rows[0])

    def where(self, condition: str) -> Frame:
        if condition == "is_current":
            kept = [r for r in self.rows if r["is_current"]]
        elif condition == "count > 1":
            kept = [r for r in self.rows if r["count"] > 1]
        else:
            nullable = re.findall(r"`(\w+)` IS NULL", condition)
            kept = [r for r in self.rows if any(r[c] is None for c in nullable)]
        return Frame(kept, self.columns)

    def count(self) -> int:
        return len(self.rows)

    def groupBy(self, *keys: str) -> Grouped:  # noqa: N802 - Spark's name
        return Grouped(self.rows, keys)


class Grouped:
    def __init__(self, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
        self.rows, self.keys = rows, keys

    def count(self) -> Frame:
        counts: dict[tuple[Any, ...], int] = {}
        for row in self.rows:
            key = tuple(row[k] for k in self.keys)
            counts[key] = counts.get(key, 0) + 1
        return Frame([{"count": n} for n in counts.values()], ["count"])


class Spark:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def table(self, name: str) -> Frame:
        return Frame(self.rows)


def run(text: str, rows: list[dict[str, Any]], tmp_path: Path) -> dict[str, dict[str, Any]]:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "test_notebook.py"
    path.write_text(text, encoding="utf-8")
    fixtures = {"spark": Spark(rows), "target_table": "customers_silver"}
    return {c["name"]: c for c in harness.run_tests(path, fixtures)}


def test_the_generated_file_passes_a_good_target(
    catalog: Catalog, make_plan: Make, tmp_path: Path
) -> None:
    rows = [{"customer_id": f"c{i}", "is_current": True} for i in range(3)]
    rows.append({"customer_id": "c0", "is_current": False})  # SCD2 history is not a duplicate
    plan = customers_plan(make_plan, strategy="scd2", incremental_column=None)
    checks = run(generate_tests(plan, catalog), rows, tmp_path)
    assert all(c["passed"] for c in checks.values()), checks
    assert len(checks) == 5 and checks["test_target_is_not_empty"]["detail"] == "passed"


def test_the_generated_file_names_what_a_bad_target_breaks(
    catalog: Catalog, make_plan: Make, tmp_path: Path
) -> None:
    rows: list[dict[str, Any]] = [
        {"customer_id": "c1"},
        {"customer_id": "c1"},
        {"customer_id": None},
    ]
    checks = run(generate_tests(customers_plan(make_plan), catalog), rows, tmp_path)
    assert checks["test_keys_are_not_null"] == {
        "name": "test_keys_are_not_null",
        "passed": False,
        "detail": "AssertionError: 1 row(s) with a null key",
    }
    assert checks["test_one_row_per_key"]["detail"] == (
        "AssertionError: 1 customer_id value(s) appear more than once"
    )
    assert checks["test_contract_customer_id_unique"]["detail"] == (
        "Failed: 1 row(s) break unique on customer_id"
    )
    assert checks["test_target_is_not_empty"]["passed"]


def test_warnings_pass_with_a_note_and_absent_columns_skip(
    catalog: Catalog, make_plan: Make, tmp_path: Path
) -> None:
    doc = catalog.docs["table:customers"]
    warned = doc.model_copy(update={"text": doc.text.replace("'fail'", "'warn'")})
    text = generate_tests(customers_plan(make_plan), Catalog([*catalog.docs.values(), warned]))
    duplicated = run(text, [{"customer_id": "c1"}, {"customer_id": "c1"}], tmp_path / "a")
    assert duplicated["test_contract_customer_id_unique"] == {
        "name": "test_contract_customer_id_unique",
        "passed": True,
        "detail": "warning: 1 row(s) break unique on customer_id",
    }
    no_column = run(text, [{"customer_name": "x"}], tmp_path / "b")
    assert no_column["test_contract_customer_id_unique"]["detail"] == (
        "skipped: customer_id is not in the target"
    )
    assert not no_column["test_keys_are_present"]["passed"]


def test_a_file_with_no_tests_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "test_notebook.py"
    empty.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pytest exited with code 5"):
        harness.run_tests(empty, {"spark": None, "target_table": "t"})

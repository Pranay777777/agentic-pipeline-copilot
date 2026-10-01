"""Planner: the plan is checked against the catalog before any code exists."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from copilot.agents.base import Catalog, Exchange, StageRejectedError, extract_json
from copilot.agents.planner import Plan, Planner, check_plan
from copilot.catalog.index import CatalogIndex
from copilot.llm import ScriptedLLM

MakePlan = Callable[..., dict[str, Any]]
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins."


def planner(catalog: Catalog, *answers: str, attempts: int = 2) -> tuple[Planner, ScriptedLLM]:
    llm = ScriptedLLM(list(answers))
    index = CatalogIndex(list(catalog.docs.values()))
    return Planner(llm, catalog, index, max_attempts=attempts), llm


def test_a_valid_plan_passes_the_catalog_check(catalog: Catalog, make_plan: MakePlan) -> None:
    plan = Plan.model_validate(make_plan())
    assert check_plan(plan, catalog) == []
    assert plan.patterns == [
        "pattern:module.ingest-incremental",
        "pattern:module.transform-silver",
        "pattern:module.ingest-cdc",
    ]


def test_names_the_lakehouse_does_not_have_are_rejected(
    catalog: Catalog, make_plan: MakePlan
) -> None:
    raw = make_plan(
        sources=["orders", "payments"],
        keys=["order_id", "order_uuid"],
        incremental_column="modified_at",
        standards=["standard:notebook.be-nice"],
    )
    raw["steps"][0]["table"] = "invoices"
    raw["steps"][1]["columns"] = ["order_id", "ts"]
    raw["steps"][2]["pattern"] = "pattern:module.magic"
    errors = check_plan(Plan.model_validate(raw), catalog)
    assert any("source 'payments' is not a catalog table" in e for e in errors)
    assert any("key 'order_uuid' is not a column" in e for e in errors)
    assert any("incremental_column 'modified_at'" in e for e in errors)
    assert (
        "step 1: table 'invoices' is neither a source (orders, payments) nor the target"
        in " ".join(errors)
    )
    assert "step 2: column 'ts' is not in orders" in errors
    assert "step 3: 'pattern:module.magic' is not a catalog pattern" in errors
    assert "'standard:notebook.be-nice' is not a catalog standard" in errors


def test_the_strategy_must_fit_the_layer(catalog: Catalog, make_plan: MakePlan) -> None:
    gold = Plan.model_validate(make_plan(strategy="scd2", layer="gold"))
    assert check_plan(gold, catalog) == ["strategy 'scd2' belongs in silver, not gold"]
    no_column = Plan.model_validate(make_plan(incremental_column=None))
    assert check_plan(no_column, catalog) == ["strategy 'dedup_latest' needs an incremental_column"]


def test_the_planner_sends_a_rejected_plan_back_with_reasons(
    catalog: Catalog, make_plan: MakePlan
) -> None:
    wrong = json.dumps(make_plan(keys=["order_uuid"]))
    right = "Here is the plan:\n```json\n" + json.dumps(make_plan()) + "\n```"
    agent, llm = planner(catalog, wrong, right)
    exchange = Exchange("plan")
    plan = agent.plan(SPEC, exchange)
    assert plan.keys == ["order_id"]
    assert [len(a.errors) for a in exchange.attempts] == [1, 0]
    feedback = llm.calls[1][-1]["content"]
    assert "key 'order_uuid' is not a column of orders" in feedback
    prompt = llm.calls[0][1]["content"]
    assert "table:orders" in prompt and "pattern:module.transform-silver" in prompt
    assert SPEC in prompt and '"strategy"' in llm.calls[0][0]["content"]


def test_the_planner_gives_up_within_its_budget(catalog: Catalog, make_plan: MakePlan) -> None:
    agent, llm = planner(catalog, "not json at all", json.dumps({"intent": "x"}))
    with pytest.raises(StageRejectedError) as caught:
        agent.plan(SPEC)
    rejected = caught.value
    assert rejected.stage == "plan" and len(rejected.attempts) == 2 and len(llm.calls) == 2
    assert "not valid JSON" in rejected.attempts[0].errors[0]
    assert any(e.startswith("layer: Field required") for e in rejected.errors)


def test_extract_json_finds_the_object() -> None:
    assert extract_json('noise {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("nothing here")


def test_a_step_may_write_the_target_using_source_columns(
    catalog: Catalog, make_plan: MakePlan
) -> None:
    raw = make_plan()
    raw["steps"][2] |= {"table": "orders_silver", "columns": ["order_id", "updated_at"]}
    assert check_plan(Plan.model_validate(raw), catalog) == []
    raw["steps"][2]["columns"] = ["order_id", "invented"]
    errors = check_plan(Plan.model_validate(raw), catalog)
    assert errors == ["step 3: column 'invented' is not in orders_silver"]


def test_the_target_may_not_overwrite_a_catalog_table(
    catalog: Catalog, make_plan: MakePlan
) -> None:
    errors = check_plan(Plan.model_validate(make_plan(target="orders")), catalog)
    assert errors == [
        "target 'orders' would overwrite a catalog table - name it for its layer, "
        "e.g. 'orders_silver'"
    ]
    assert check_plan(Plan.model_validate(make_plan(target="customers")), catalog)

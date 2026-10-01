"""Generator: a Databricks notebook whose every cell cites what it follows."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from typing import Any

import pytest

from copilot.agents.base import Catalog, Exchange, StageRejectedError
from copilot.agents.generator import Generator, NotebookDraft, check_draft, render
from copilot.agents.planner import Plan
from copilot.llm import ScriptedLLM

Make = Callable[..., dict[str, Any]]


def test_the_notebook_is_databricks_source_that_parses(make_plan: Make, make_draft: Make) -> None:
    plan = Plan.model_validate(make_plan())
    text = render(NotebookDraft.model_validate(make_draft()), plan)
    assert text.startswith("# Databricks notebook source\n# MAGIC %md\n# MAGIC # orders_silver")
    assert text.count("# COMMAND ----------") == 4  # intro, parameters, three cells
    assert 'dbutils.widgets.text("orders_table", "")' in text
    assert 'target_table = dbutils.widgets.get("target_table")' in text
    assert "# DBTITLE 1,MERGE into Silver\n# Follows: pattern:module.ingest-cdc" in text
    ast.parse(text)
    no_params = render(NotebookDraft.model_validate(make_draft(parameters=[])), plan)
    assert "DBTITLE 1,Parameters" not in no_params


def test_a_good_draft_passes(catalog: Catalog, make_plan: Make, make_draft: Make) -> None:
    plan = Plan.model_validate(make_plan())
    assert check_draft(NotebookDraft.model_validate(make_draft()), plan, catalog) == []


def test_drafts_that_drift_from_the_plan_are_rejected(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    raw = make_draft(parameters=["Table", "x", "x"])
    raw["cells"][0]["code"] = "def broken(:\n    pass"
    raw["cells"][1]["cites"] = ["pattern:module.transform-scd2", "pattern:module.nope"]
    raw["cells"][2]["cites"] = ["table:customers"]
    errors = check_draft(
        NotebookDraft.model_validate(raw), Plan.model_validate(make_plan()), catalog
    )
    assert (
        "parameter 'Table' is not in the contract "
        "(orders_table, target_table, watermark, key_columns)" in errors
    )
    assert "parameters repeat a name" in errors
    assert any(e.startswith("cell 1 'Read new Bronze rows' line 1:") for e in errors)
    assert "cell 2 cites 'pattern:module.transform-scd2', which the plan did not choose" in errors
    assert "cell 2 cites 'pattern:module.nope', which is not in the catalog" in errors
    assert "cell 3 cites 'table:customers', which the plan did not choose" in errors
    assert "no cell follows the plan's 'pattern:module.transform-silver'" in errors
    assert "no cell follows the plan's 'pattern:module.ingest-cdc'" in errors


def test_the_generator_retries_then_revises_on_review(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    plan = Plan.model_validate(make_plan())
    uncited = make_draft()
    uncited["cells"] = uncited["cells"][:1]
    llm = ScriptedLLM([json.dumps(uncited), json.dumps(make_draft()), json.dumps(make_draft())])
    agent = Generator(llm, catalog, max_attempts=2)
    exchange = Exchange("generate")
    draft = agent.generate(plan, exchange)
    assert len(draft.cells) == 3 and [len(a.errors) for a in exchange.attempts] == [2, 0]
    prompt = llm.calls[0][1]["content"]
    assert '"target": "orders_silver"' in prompt and "### table:orders" in prompt
    assert "### standard:notebook.writes-are-idempotent" in prompt
    agent.generate(plan, review=["line 9: blind append"], previous=draft)
    revision = llm.calls[2][1]["content"]
    assert "# Your previous notebook" in revision and "- line 9: blind append" in revision


def test_the_generator_gives_up_within_its_budget(catalog: Catalog, make_plan: Make) -> None:
    agent = Generator(ScriptedLLM(["{}"]), catalog, max_attempts=1)
    with pytest.raises(StageRejectedError) as caught:
        agent.generate(Plan.model_validate(make_plan()))
    assert caught.value.stage == "generate" and "cells: Field required" in caught.value.errors

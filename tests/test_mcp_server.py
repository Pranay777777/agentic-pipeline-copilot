"""MCP tool layer: what a client can see and call, in-process through the real MCP client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
from mcp import Client

from copilot.agents.base import Catalog
from copilot.agents.generator import NotebookDraft, render
from copilot.agents.planner import Plan
from copilot.mcp_server import MAX_NOTEBOOK, build_server
from copilot.sandbox.runner import DockerSandbox

Make = Callable[..., dict[str, Any]]


def call(server: Any, tool: str, arguments: dict[str, Any]) -> tuple[bool, Any]:
    """(is_error, structured result or error text) for one tool call."""

    async def go() -> tuple[bool, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, arguments)
        if result.is_error:
            return True, result.content[0].text  # type: ignore[union-attr]
        content = result.structured_content or {}
        return False, content.get("result", content)

    return anyio.run(go)


def tools(server: Any) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(server) as client:
            listed = await client.list_tools()
        return {t.name: t.annotations for t in listed.tools}

    return anyio.run(go)


def test_only_lookup_and_gates_are_exposed_never_a_model_or_a_write(catalog: Catalog) -> None:
    found = tools(build_server(catalog))
    assert set(found) == {
        "search_catalog",
        "get_catalog_doc",
        "review_notebook",
        "generate_notebook_tests",
        "validate_notebook",
    }
    for name, hints in found.items():
        assert hints.open_world_hint is False
        assert hints.read_only_hint is (name != "validate_notebook")
        assert not hints.destructive_hint


def test_catalog_search_and_lookup(catalog: Catalog) -> None:
    server = build_server(catalog)
    error, hits = call(server, "search_catalog", {"query": "orders table", "kind": "table"})
    assert not error and hits[0]["id"] == "table:orders"
    assert all(h["kind"] == "table" for h in hits)
    error, many = call(server, "search_catalog", {"query": "silver merge", "limit": 500})
    assert not error and 0 < len(many) <= 20
    error, doc = call(server, "get_catalog_doc", {"doc_id": "table:customers"})
    assert not error and "Expect:" in doc["text"] and doc["kind"] == "table"
    assert call(server, "get_catalog_doc", {"doc_id": "table:nope"}) == (
        True,
        "Error executing tool get_catalog_doc: no catalog document 'table:nope'",
    )
    error, text = call(server, "search_catalog", {"query": "x", "kind": "secrets"})
    assert error and "kind must be table, pattern or standard" in text
    error, text = call(server, "search_catalog", {"query": "x" * 501})
    assert error and "longer than 500" in text


def test_review_and_tests_need_no_model(catalog: Catalog, make_plan: Make) -> None:
    server = build_server(catalog)
    error, verdict = call(server, "review_notebook", {"notebook": "rows = df.collect()\n"})
    assert not error and not verdict["passed"] and "collect()" in verdict["findings"][0]
    error, text = call(server, "review_notebook", {"notebook": "x" * (MAX_NOTEBOOK + 1)})
    assert error and "the limit is 200000" in text
    error, tests = call(server, "generate_notebook_tests", {"plan": make_plan()})
    assert not error and "def test_one_row_per_key" in tests
    error, text = call(server, "generate_notebook_tests", {"plan": make_plan(target="orders")})
    assert error and "would overwrite a catalog table" in text
    error, text = call(server, "generate_notebook_tests", {"plan": {"intent": "x"}})
    assert error and "not a valid plan" in text


def test_validation_runs_only_in_the_sandbox(
    catalog: Catalog,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: dict[str, Any],
) -> None:
    plan = make_plan()
    notebook = render(NotebookDraft.model_validate(make_draft()), Plan.model_validate(plan))
    sandbox = sandbox_factory(results["passed"])
    error, report = call(
        build_server(catalog, sandbox), "validate_notebook", {"plan": plan, "notebook": notebook}
    )
    assert not error and report["passed"] and report["errors"] == []
    assert {t["name"] for t in report["tests"]} >= {"test_one_row_per_key"}
    assert sandbox.notebooks == [notebook]
    arguments = {"plan": plan, "notebook": notebook}
    error, text = call(build_server(catalog), "validate_notebook", arguments)
    assert error and "no sandbox" in text
    missing = DockerSandbox("no-such-image:0", docker="no-such-docker-binary")
    error, text = call(build_server(catalog, missing), "validate_notebook", arguments)
    assert error and "'no-such-image:0' not found" in text

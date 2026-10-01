"""The real sandbox: isolation, a Delta MERGE, the static gate, a broken notebook, the time limit.

Needs Docker and the image (`python -m copilot.sandbox build`); skipped unless
COPILOT_SANDBOX_IMAGE names it. CI's sandbox job builds the image and runs these.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from copilot.agents.base import Catalog
from copilot.agents.generator import NotebookDraft, render
from copilot.agents.planner import Plan
from copilot.agents.validator import Validator
from copilot.sandbox.__main__ import isolation
from copilot.sandbox.runner import DockerSandbox, Limits

IMAGE = os.environ.get("COPILOT_SANDBOX_IMAGE", "")
pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not IMAGE, reason="set COPILOT_SANDBOX_IMAGE to run the real sandbox"),
]
Make = Callable[..., dict[str, Any]]


def notebook(make_plan: Make, draft: dict[str, Any]) -> tuple[Plan, str]:
    plan = Plan.model_validate(make_plan())
    return plan, render(NotebookDraft.model_validate(draft), plan)


def test_every_escape_is_blocked() -> None:
    ok, found = isolation(DockerSandbox(IMAGE, Limits(timeout_s=60)))
    assert ok, found
    assert found["network"].startswith("blocked") and found["uid"] == 1000


def test_a_delta_merge_notebook_runs_and_meets_the_plan(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    plan, text = notebook(make_plan, make_draft())
    report = Validator(DockerSandbox(IMAGE), catalog).validate(plan, text)
    assert report.passed, (report.errors, report.log_tail)
    assert len(report.ran) == 4
    checks = {c["name"]: c for c in report.checks}
    assert set(checks) == {
        "test_target_is_not_empty",
        "test_keys_are_present",
        "test_keys_are_not_null",
        "test_one_row_per_key",
    }
    assert all(c["passed"] for c in checks.values())


def test_a_wrong_column_comes_back_with_its_notebook_line(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    draft = make_draft()
    draft["cells"][0]["code"] = draft["cells"][0]["code"].replace(
        '"order_status", ', '"order_ts", '
    )
    plan, text = notebook(make_plan, draft)
    report = Validator(DockerSandbox(IMAGE), catalog).validate(plan, text)
    error = report.errors[0]
    assert (error.stage, error.cell, error.kind) == ("cell", 2, "AnalysisException")
    assert error.line is not None and "order_ts" in error.message
    assert "select(*columns)" in text.splitlines()[error.line - 1]


def test_a_notebook_that_does_not_finish_is_killed(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    draft = make_draft()
    draft["cells"][0]["code"] = "import time\n\ntime.sleep(600)"
    plan, text = notebook(make_plan, draft)
    report = Validator(DockerSandbox(IMAGE, Limits(timeout_s=20)), catalog).validate(plan, text)
    assert report.timed_out and str(report.errors[0]).startswith("sandbox: Timeout")


def test_a_static_finding_stops_the_notebook_before_spark(
    catalog: Catalog, make_plan: Make, make_draft: Make
) -> None:
    draft = make_draft()
    draft["cells"][1]["code"] = draft["cells"][1]["code"].replace(
        "new_rows.withColumn", "new_row.withColumn"
    )
    plan, text = notebook(make_plan, draft)
    report = Validator(DockerSandbox(IMAGE), catalog).validate(plan, text)
    assert report.ran == () and report.checks == ()
    kinds = {(e.stage, e.kind) for e in report.errors}
    assert ("static", "ruff F821") in kinds and ("static", "mypy name-defined") in kinds
    line = report.errors[0].line
    assert line is not None and "new_row.withColumn" in text.splitlines()[line - 1]

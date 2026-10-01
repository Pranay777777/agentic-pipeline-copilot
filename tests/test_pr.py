"""Pull-request tool: one repository, a new branch, an opened PR - and never a merge."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from copilot.agents.__main__ import main as agents_main
from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline, Run
from copilot.agents.validator import Validator
from copilot.config import get_settings
from copilot.llm import ScriptedLLM
from copilot.pr import GitHubPR, PRError, branch_name, files_for, main, pr_body

Make = Callable[..., dict[str, Any]]
Results = dict[str, dict[str, Any]]
REPO = "Pranay777777/copilot-playground"
TOKEN = "fake"
SPEC = "Load orders incrementally into Silver, one row per order_id, newest wins."
ROOT = Path(__file__).resolve().parents[1]


class GitHub:
    """A scripted api.github.com that records every request."""

    def __init__(self, existing: dict[str, str] | None = None, fail: str | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.existing = existing or {}
        self.fail = fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if self.fail and self.fail in path:
            return httpx.Response(422, json={"message": "Reference already exists"})
        if path.endswith("/git/ref/heads/main"):
            return httpx.Response(200, json={"object": {"sha": "base-sha"}})
        if path.endswith("/git/refs"):
            return httpx.Response(201, json={})
        if "/contents/" in path and method == "GET":
            name = path.split("/contents/", 1)[1]
            if name in self.existing:
                return httpx.Response(200, json={"sha": self.existing[name]})
            return httpx.Response(404, json={"message": "Not Found"})
        if "/contents/" in path:
            return httpx.Response(201, json={})
        if path.endswith("/pulls") and method == "POST":
            url = f"https://github.com/{REPO}/pull/7"
            return httpx.Response(201, json={"html_url": url, "number": 7})
        if path.endswith("/pulls"):
            return httpx.Response(200, json=[])
        if path == f"/repos/{REPO}":
            headers = {"github-authentication-token-expiration": "2026-12-30 00:00:00 UTC"}
            body = {"full_name": REPO, "visibility": "public", "default_branch": "main"}
            return httpx.Response(200, json=body, headers=headers)
        return httpx.Response(404, json={"message": "Not Found"})  # pragma: no cover

    def client(self) -> GitHubPR:
        return GitHubPR(TOKEN, REPO, transport=httpx.MockTransport(self))


FILES = {
    "notebooks/orders_silver.py": "x = 1\n",
    "tests/test_orders_silver.py": "def test(): ...\n",
}


def test_open_branches_commits_each_file_and_opens_the_pr() -> None:
    github = GitHub(existing={"tests/test_orders_silver.py": "old-sha"})
    pull = github.client().open("copilot/orders_silver-20261001120000", FILES, "title", "body")
    assert (pull.url, pull.number) == (f"https://github.com/{REPO}/pull/7", 7)
    calls = [(r.method, r.url.path.removeprefix(f"/repos/{REPO}")) for r in github.requests]
    assert calls == [
        ("GET", "/git/ref/heads/main"),
        ("POST", "/git/refs"),
        ("GET", "/contents/notebooks/orders_silver.py"),
        ("PUT", "/contents/notebooks/orders_silver.py"),
        ("GET", "/contents/tests/test_orders_silver.py"),
        ("PUT", "/contents/tests/test_orders_silver.py"),
        ("POST", "/pulls"),
    ]
    sent = [json.loads(r.content) if r.content else {} for r in github.requests]
    assert sent[1] == {"ref": "refs/heads/copilot/orders_silver-20261001120000", "sha": "base-sha"}
    assert base64.b64decode(sent[3]["content"]).decode() == "x = 1\n" and "sha" not in sent[3]
    assert sent[5]["sha"] == "old-sha" and sent[5]["branch"].startswith("copilot/")
    assert sent[6]["maintainer_can_modify"] is False
    assert (sent[6]["head"], sent[6]["base"]) == ("copilot/orders_silver-20261001120000", "main")
    assert github.requests[2].url.params["ref"] == "copilot/orders_silver-20261001120000"
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in github.requests)


def test_it_only_ever_touches_its_one_repository_and_cannot_merge() -> None:
    github = GitHub()
    github.client().open("copilot/x-1", FILES, "t", "b")
    github.client().check()
    for request in github.requests:
        assert request.url.host == "api.github.com"
        assert request.url.path.startswith(f"/repos/{REPO}")
        assert "merge" not in request.url.path
    assert not any("merge" in name for name in dir(GitHubPR))


def test_refusals_say_what_failed_and_never_the_token() -> None:
    with pytest.raises(
        PRError, match=r"POST /repos/.*/git/refs: 422 Reference already exists"
    ) as e:
        GitHub(fail="/git/refs").client().open("copilot/x-1", FILES, "t", "b")
    assert TOKEN not in str(e.value)

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(PRError, match="ConnectError") as e:
        GitHubPR(TOKEN, REPO, transport=httpx.MockTransport(offline)).check()
    assert TOKEN not in str(e.value)

    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    with pytest.raises(PRError, match="502 <html>bad gateway</html>"):
        GitHubPR(TOKEN, REPO, transport=httpx.MockTransport(html)).check()
    with pytest.raises(PRError, match="COPILOT_GITHUB_TOKEN"):
        GitHubPR("", REPO)
    with pytest.raises(PRError, match="owner/name"):
        GitHubPR(TOKEN, "https://github.com/someone/else")


def test_branch_names_are_per_target_and_utc() -> None:
    when = datetime(2026, 10, 1, 9, 5, 7, tzinfo=UTC)
    assert branch_name("orders_silver", when) == "copilot/orders_silver-20261001090507"
    assert branch_name("x").startswith("copilot/x-20")


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PR_REPO", REPO)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", TOKEN)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_check_reports_access_and_expiry(
    configured: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check"], transport=httpx.MockTransport(GitHub())) == 0
    out = capsys.readouterr().out
    assert f"repo            {REPO}" in out and "visibility      public" in out
    assert "token expires   2026-12-30 00:00:00 UTC" in out and "never merges" in out
    configured.setenv("PR_REPO", "")
    get_settings.cache_clear()
    assert main(["check"]) == 1
    assert "PR_REPO is not set" in capsys.readouterr().err


# --- the PR's content and the CLI --------------------------------------------------


def ready_run(catalog: Catalog, make_plan: Make, make_draft: Make, sandbox: Any) -> Run:
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    return Pipeline(llm, catalog, validator=Validator(sandbox, catalog)).run(SPEC)


def test_the_body_carries_the_evidence_and_the_files_ship_together(
    catalog: Catalog, make_plan: Make, make_draft: Make, sandbox_factory: Any, results: Results
) -> None:
    run = ready_run(catalog, make_plan, make_draft, sandbox_factory(results["passed"]))
    body = pr_body(run)
    assert body.startswith(f"**Spec:** {SPEC}") and "dedup_latest into silver" in body
    assert "1. Read new Bronze orders since the watermark" in body
    assert "- `pattern:module.ingest-cdc`" in body and "- `table:orders`" in body
    assert "- Static gate (ruff, mypy): no finding(s)" in body
    assert "- pass `test_one_row_per_key` - passed" in body
    assert "Model calls: 2, tokens: 30" in body and "a human reviews and merges" in body
    assert set(files_for(run)) == {"notebooks/orders_silver.py", "tests/test_orders_silver.py"}


def test_cli_opens_a_pr_only_for_a_ready_notebook(
    configured: pytest.MonkeyPatch,
    tmp_path: Path,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: Results,
    capsys: pytest.CaptureFixture[str],
) -> None:
    github = GitHub()
    out = tmp_path / "orders_silver.py"
    argv = ["run", SPEC, "--out", str(out), "--open-pr"]
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    sandbox = sandbox_factory(results["passed"])
    assert agents_main(argv, llm=llm, sandbox=sandbox, github=github.client()) == 0
    assert f"pull request: https://github.com/{REPO}/pull/7" in capsys.readouterr().out
    pulls = [r for r in github.requests if r.url.path.endswith("/pulls")]
    assert json.loads(pulls[0].content)["title"] == "copilot: orders_silver (dedup_latest, silver)"

    escalated = GitHub()
    llm = ScriptedLLM([json.dumps(make_plan())] + [json.dumps(make_draft())] * 3)
    sandbox = sandbox_factory(*[results["duplicates"]] * 3)
    argv = [*argv, "--report", str(tmp_path / "why.md")]
    assert agents_main(argv, llm=llm, sandbox=sandbox, github=escalated.client()) == 1
    assert escalated.requests == []


def test_cli_checks_the_pr_config_before_any_model_call(
    configured: pytest.MonkeyPatch,
    make_plan: Make,
    make_draft: Make,
    sandbox_factory: Any,
    results: Results,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = ScriptedLLM([])
    assert agents_main(["run", SPEC, "--open-pr", "--no-sandbox"], llm=llm) == 2
    assert "only ready notebooks" in capsys.readouterr().err
    configured.setenv("COPILOT_GITHUB_TOKEN", "")
    get_settings.cache_clear()
    assert agents_main(["run", SPEC, "--open-pr"], llm=llm, sandbox=sandbox_factory()) == 2
    assert "COPILOT_GITHUB_TOKEN is not set" in capsys.readouterr().err
    assert llm.calls == []
    refused = GitHub(fail="/git/refs").client()
    llm = ScriptedLLM([json.dumps(make_plan()), json.dumps(make_draft())])
    argv = ["run", SPEC, "--out", str(tmp_path / "nb.py"), "--open-pr"]
    sandbox = sandbox_factory(results["passed"])
    assert agents_main(argv, llm=llm, sandbox=sandbox, github=refused) == 2
    assert "pull request not opened: GitHub POST" in capsys.readouterr().err
    assert (tmp_path / "nb.py").exists()

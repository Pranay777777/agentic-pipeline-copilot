"""Pull-request tool (step 74): open a PR with a ready notebook and its tests - never merge.

Least privilege by construction (ADR-006):

- it talks to **one repository**, `PR_REPO`, through a fine-grained token
  (`COPILOT_GITHUB_TOKEN`) scoped to that repository with Contents and Pull
  requests write - nothing else;
- every request goes to `/repos/<PR_REPO>/...`; there is no code path to
  another repository, to settings, or to the merge endpoint;
- it only ever writes to a new `copilot/<target>-<UTC timestamp>` branch,
  opens the PR with `maintainer_can_modify` off, and stops: **a human reviews
  and merges**;
- the token is never logged or put in an error message.

    python -m copilot.pr check     # can the token see PR_REPO? when does it expire?
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from copilot.config import get_settings

if TYPE_CHECKING:
    from copilot.agents.pipeline import Run

API = "https://api.github.com"
REPO = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")
EXPIRY_HEADER = "github-authentication-token-expiration"


class PRError(Exception):
    """GitHub refused or the configuration is incomplete; never contains the token."""


@dataclass(frozen=True)
class PullRequest:
    url: str
    number: int
    branch: str


def branch_name(target: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M%S")
    return f"copilot/{target}-{stamp}"


class GitHubPR:
    """Opens pull requests on one repository. Deliberately has no merge method."""

    def __init__(
        self,
        token: str,
        repo: str,
        base: str = "main",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not token:
            raise PRError("COPILOT_GITHUB_TOKEN is not set - a fine-grained token for PR_REPO")
        if not REPO.match(repo):
            raise PRError(f"PR_REPO must look like owner/name, got {repo!r}")
        self.repo, self.base = repo, base
        self._client = httpx.Client(
            base_url=API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "agentic-pipeline-copilot",
            },
            transport=transport,
            timeout=timeout,
        )

    def _call(
        self, method: str, path: str, ok: tuple[int, ...] = (200, 201), **kwargs: Any
    ) -> httpx.Response:
        url = f"/repos/{self.repo}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise PRError(f"GitHub {method} {url}: {type(exc).__name__}") from None
        if response.status_code not in ok:
            try:
                message = str(response.json().get("message", ""))
            except ValueError:
                message = response.text[:200]
            raise PRError(f"GitHub {method} {url}: {response.status_code} {message}".strip())
        return response

    def open(self, branch: str, files: Mapping[str, str], title: str, body: str) -> PullRequest:
        """Branch from the base, commit each file to the branch, open the PR."""
        head = self._call("GET", f"/git/ref/heads/{quote(self.base)}").json()
        self._call(
            "POST", "/git/refs", json={"ref": f"refs/heads/{branch}", "sha": head["object"]["sha"]}
        )
        for path, text in files.items():
            route = f"/contents/{quote(path)}"
            existing = self._call("GET", route, ok=(200, 404), params={"ref": branch})
            payload = {
                "message": f"copilot: add {path}",
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                "branch": branch,
            }
            if existing.status_code == 200:
                payload["sha"] = existing.json()["sha"]
            self._call("PUT", route, json=payload)
        pull = self._call(
            "POST",
            "/pulls",
            json={
                "title": title,
                "head": branch,
                "base": self.base,
                "body": body,
                "maintainer_can_modify": False,
            },
        ).json()
        return PullRequest(pull["html_url"], int(pull["number"]), branch)

    def check(self) -> dict[str, str]:
        """What the token can see of the repository, and when it expires."""
        response = self._call("GET", "")
        repo = response.json()
        self._call("GET", "/pulls", params={"per_page": 1})
        return {
            "repo": str(repo.get("full_name", self.repo)),
            "visibility": str(repo.get("visibility", "unknown")),
            "default branch": str(repo.get("default_branch", "?")),
            "pull requests": "readable",
            "token expires": response.headers.get(EXPIRY_HEADER, "never (or not reported)"),
        }


def files_for(run: Run) -> dict[str, str]:
    assert run.plan is not None and run.notebook is not None and run.tests is not None
    target = run.plan.target
    return {f"notebooks/{target}.py": run.notebook, f"tests/test_{target}.py": run.tests}


def pr_body(run: Run) -> str:
    """The PR description: what was asked, what was planned, and the evidence it is ready."""
    assert run.plan is not None
    plan = run.plan
    cited = sorted({c for cell in run.draft.cells for c in cell.cites} if run.draft else set())
    last = run.rounds[-1] if run.rounds else None
    lines = [
        f"**Spec:** {run.spec}",
        "",
        "## Plan",
        "",
        f"{plan.strategy} into {plan.layer} `{plan.target}` from "
        f"{', '.join(f'`{s}`' for s in plan.sources)}; keys {', '.join(plan.keys)}.",
        "",
        *[f"{n}. {step.action}" for n, step in enumerate(plan.steps, 1)],
        "",
        "## Catalog patterns and standards cited",
        "",
        *[f"- `{c}`" for c in cited or plan.patterns],
        *[f"- `{s}`" for s in plan.standards if s not in cited],
        "",
        "## Evidence",
        "",
    ]
    if last is not None:
        lines.append(f"- Review (Critic): {'passed' if last.review.passed else 'findings left'}")
        if last.validation is not None:
            static = last.validation.static
            lines.append(f"- Static gate (ruff, mypy): {len(static) or 'no'} finding(s)")
            ran = len(last.validation.ran)
            lines.append(
                f"- Sandbox execution: {ran} cell(s) on sample data, "
                f"{last.validation.seconds:.0f}s, no network"
            )
            lines += [
                f"- {'pass' if c['passed'] else 'FAIL'} `{c['name']}` - {c['detail']}"
                for c in last.validation.checks
            ]
    lines += [
        f"- Correction rounds: {run.corrections}",
        f"- Model calls: {len(run.attempts)}, tokens: {run.tokens}",
        "",
        "---",
        "Opened by agentic-pipeline-copilot. It never merges: **a human reviews and merges.**",
    ]
    return "\n".join(lines) + "\n"


def from_settings(transport: httpx.BaseTransport | None = None) -> GitHubPR:
    settings = get_settings()
    if not settings.pr_repo:
        raise PRError("PR_REPO is not set - e.g. PR_REPO=owner/copilot-playground")
    token = settings.copilot_github_token.get_secret_value()
    return GitHubPR(token, settings.pr_repo, settings.pr_base_branch, transport)


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.pr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["check"])
    parser.parse_args(argv)
    try:
        found = from_settings(transport).check()
    except PRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for key, value in found.items():
        print(f"{key:<15} {value}")
    print("ok - write access is proven by the first pull request (the copilot never merges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Agents CLI.

    python -m copilot.agents run "Load orders incrementally into Silver, newest row per order"
        [--out notebooks/orders_silver.py] [--trace runs/orders.json] [--no-sandbox] [--open-pr]
        [--record runs/orders.cassette.jsonl | --replay runs/orders.cassette.jsonl]
    python -m copilot.agents plan "..."
    python -m copilot.agents review notebooks/orders_silver.py

`run` and `plan` call the model (OPENROUTER_API_KEY, LLM_MODEL - ADR-004).
`run` executes the notebook in the sandbox (Docker, image built with
`python -m copilot.sandbox build` - ADR-005); `--no-sandbox` stops after
review and the result is "reviewed", never "ready". A ready notebook is
written with its generated tests (`test_<target>.py`); `--open-pr` also opens
a pull request on PR_REPO with both (COPILOT_GITHUB_TOKEN - ADR-006) - it
never merges. `--record` writes every model call to a cassette; `--replay` answers
from one with no model, no key and no network (ADR-007). `review` runs the
Critic's rules on any notebook and needs no key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.agents.base import Attempt, Catalog, StageRejectedError
from copilot.agents.critic import review
from copilot.agents.pipeline import Pipeline, diagnostic_report
from copilot.agents.validator import Validator
from copilot.catalog.model import read_snapshot
from copilot.config import Settings, get_settings
from copilot.llm import LLM, LLMError, OpenRouterLLM
from copilot.pr import GitHubPR, PRError, branch_name, files_for, from_settings, pr_body
from copilot.replay import RecordingLLM, ReplayLLM
from copilot.sandbox.runner import DockerSandbox, Limits, Sandbox

SNAPSHOT = Path("catalog/snapshot.jsonl")


def progress(attempt: Attempt) -> None:
    verdict = "ok" if not attempt.errors else f"sent back ({len(attempt.errors)} problem(s))"
    tokens = attempt.prompt_tokens + attempt.completion_tokens
    print(
        f"  {attempt.stage} #{attempt.number}: {verdict} - {attempt.model}, {tokens} tokens",
        file=sys.stderr,
        flush=True,
    )


def make_llm(settings: Settings) -> LLM:
    key = settings.llm_api_key
    if not key.get_secret_value() and settings.llm_provider == "openrouter":
        key = settings.openrouter_api_key  # never sent to any other provider
    return OpenRouterLLM(
        key,
        settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort or None,
        fallbacks=[m.strip() for m in settings.llm_fallback_models.split(",")],
        max_retries=settings.llm_max_retries,
        provider=settings.llm_provider,
    )


def make_sandbox(settings: Settings) -> DockerSandbox:
    limits = Limits(
        cpus=settings.sandbox_cpus,
        memory=settings.sandbox_memory,
        timeout_s=settings.sandbox_timeout_s,
    )
    return DockerSandbox(settings.sandbox_image, limits)


def main(
    argv: list[str] | None = None,
    llm: LLM | None = None,
    sandbox: Sandbox | None = None,
    github: GitHubPR | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.agents",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="plan, generate, review and execute one notebook")
    run.add_argument("spec")
    run.add_argument("--out", type=Path, help="write the notebook here when it is ready")
    run.add_argument("--trace", type=Path, help="write the run's trajectory as JSON")
    run.add_argument("--report", type=Path, help="where to write the report if it escalates")
    run.add_argument("--no-sandbox", action="store_true", help="stop after review (not ready)")
    run.add_argument(
        "--open-pr", action="store_true", help="open a pull request on PR_REPO when ready"
    )
    tape = run.add_mutually_exclusive_group()
    tape.add_argument("--record", type=Path, help="write every model call to this cassette")
    tape.add_argument("--replay", type=Path, help="answer from this cassette; no model")
    plan = sub.add_parser("plan", help="plan only, checked against the catalog")
    plan.add_argument("spec")
    check = sub.add_parser("review", help="run the Critic's rules on a notebook")
    check.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    if args.command == "review":
        verdict = review(args.path.read_text(encoding="utf-8"))
        for finding in verdict.findings:
            print(f"{args.path}:{finding}")
        print("passed" if verdict.passed else f"{len(verdict.findings)} finding(s)")
        return 0 if verdict.passed else 1

    settings = get_settings()
    if args.command == "run" and args.open_pr:
        if args.no_sandbox:
            print("error: --open-pr needs the sandbox: only ready notebooks", file=sys.stderr)
            return 2
        try:  # checked before any model call, so a bad config costs nothing
            github = github or from_settings()
        except PRError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    catalog = Catalog(read_snapshot(SNAPSHOT))
    validator = None
    if args.command == "run" and not args.no_sandbox:
        sandbox = sandbox or make_sandbox(settings)
        if isinstance(sandbox, DockerSandbox) and not sandbox.available():
            print(
                f"error: sandbox image '{settings.sandbox_image}' not found - start Docker and "
                "run `python -m copilot.sandbox build` (or pass --no-sandbox)",
                file=sys.stderr,
            )
            return 2
        validator = Validator(sandbox, catalog)
    replay = args.command == "run" and args.replay is not None
    try:
        if replay:
            llm = llm or ReplayLLM(args.replay)
        llm = llm or make_llm(settings)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "run" and args.record is not None:
        llm = RecordingLLM(llm, args.record)
    pipeline = Pipeline(
        llm, catalog, settings.agent_max_attempts, settings.correction_max_rounds, validator
    )
    try:
        if args.command == "plan":
            print(pipeline.planner.plan(args.spec).model_dump_json(indent=2))
            return 0
        note = (
            f"replaying {args.replay} - no model, no network"
            if replay
            else f"running with {settings.llm_model} - free models can take minutes..."
        )
        print(note, file=sys.stderr, flush=True)
        result = pipeline.run(args.spec, on_attempt=progress)
    except StageRejectedError as exc:
        print(f"rejected at {exc.stage}:", *exc.errors, sep="\n- ", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    calls = (
        f"{len(result.attempts)} model call(s), {result.tokens} tokens, "
        f"{result.corrections} correction(s)"
    )
    if result.status in ("rejected", "escalated"):
        target = result.plan.target if result.plan else "run"
        report = args.report or Path("runs") / f"{target}.report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(diagnostic_report(result), encoding="utf-8")
        reasons = "\n- ".join(result.reasons)
        print(f"{result.status} at {result.stage} ({calls}):\n- {reasons}", file=sys.stderr)
        print(f"diagnostic report: {report}", file=sys.stderr)
        return 1
    assert result.notebook is not None and result.plan is not None
    out = args.out or Path("notebooks") / f"{result.plan.target}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.notebook, encoding="utf-8", newline="\n")
    written = str(out)
    if result.tests is not None:
        tests = out.parent / f"test_{result.plan.target}.py"
        tests.write_text(result.tests, encoding="utf-8", newline="\n")
        written += f" + {tests}"
    print(f"{result.status}: {written} ({calls})")
    if github is not None and result.status == "ready":
        title = f"copilot: {result.plan.target} ({result.plan.strategy}, {result.plan.layer})"
        try:
            pull = github.open(
                branch_name(result.plan.target), files_for(result), title, pr_body(result)
            )
        except PRError as exc:
            print(f"error: pull request not opened: {exc}", file=sys.stderr)
            return 2
        print(f"pull request: {pull.url} (a human reviews and merges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

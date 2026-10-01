"""Agents CLI.

    python -m copilot.agents run "Load orders incrementally into Silver, newest row per order"
        [--out notebooks/orders_silver.py] [--trace runs/orders.json]
    python -m copilot.agents plan "..."
    python -m copilot.agents review notebooks/orders_silver.py

`run` and `plan` call the model (OPENROUTER_API_KEY, LLM_MODEL - ADR-004);
`review` runs the Critic's rules on any notebook and needs no key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.agents.base import Catalog, StageRejectedError
from copilot.agents.critic import review
from copilot.agents.pipeline import Pipeline
from copilot.catalog.model import read_snapshot
from copilot.config import Settings, get_settings
from copilot.llm import LLM, LLMError, OpenRouterLLM

SNAPSHOT = Path("catalog/snapshot.jsonl")


def make_llm(settings: Settings) -> LLM:
    return OpenRouterLLM(
        settings.openrouter_api_key,
        settings.llm_model,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_s,
        max_tokens=settings.llm_max_tokens,
    )


def main(argv: list[str] | None = None, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.agents",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="plan, generate and review one notebook")
    run.add_argument("spec")
    run.add_argument("--out", type=Path, help="write the notebook here when it is ready")
    run.add_argument("--trace", type=Path, help="write the run's attempts and reviews as JSON")
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
    catalog = Catalog(read_snapshot(SNAPSHOT))
    try:
        llm = llm or make_llm(settings)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    pipeline = Pipeline(llm, catalog, settings.agent_max_attempts, settings.critic_max_rounds)
    try:
        if args.command == "plan":
            print(pipeline.planner.plan(args.spec).model_dump_json(indent=2))
            return 0
        result = pipeline.run(args.spec)
    except StageRejectedError as exc:
        print(f"rejected at {exc.stage}:", *exc.errors, sep="\n- ", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")
    calls = f"{len(result.attempts)} model call(s), {result.tokens} tokens"
    if result.status == "rejected":
        print(
            f"rejected at {result.stage} ({calls}):", *result.reasons, sep="\n- ", file=sys.stderr
        )
        return 1
    assert result.notebook is not None and result.plan is not None
    out = args.out or Path("notebooks") / f"{result.plan.target}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.notebook, encoding="utf-8", newline="\n")
    print(f"ready: {out} ({calls})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

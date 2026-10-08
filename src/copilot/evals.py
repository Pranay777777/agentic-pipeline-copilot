"""Agent eval suite (78), its CI gate (79) and its published images (80).

    python -m copilot.evals record --limit 6     # live: record up to 6 unrecorded specs
    python -m copilot.evals run                  # replay every recorded spec; write the report
    python -m copilot.evals gate                 # run, then fail on regressions or a low rate
    python -m copilot.evals gate --update-baseline
    python -m copilot.evals render               # docs/images/*.svg from the report

`evals/specs.json` holds natural-language specs with the plan each should
produce (strategy, layer, sources, keys - only what the spec pins down). A
spec passes when the run ends `reviewed` and its plan matches.

Recording is live and slow on free tiers (about 20 requests a day), so it
is done in batches: `record` skips specs that already have a cassette. A
transient provider error (timeout, 429, 5xx) is retried after 5 s and 15 s; if
the spec still fails it is left "pending (transient)" - no cassette, not an eval
failure - and recording moves on after a 45 s pause. That backoff is the only
retry layer: the client is built with its own retries off, so each attempt is
exactly one HTTP request against the daily quota. A 429 saying the daily quota
is used up stops the batch at once: no retry, nothing marked pending. Any other
error stops the batch too, keeping what it recorded. Everything else
replays cassettes - no model, no key, no network - so the gate runs in CI on
every push for free (ADR-008).

The gate fails when a spec that passed in `evals/baseline.json` now fails (a
regression), or when the pass rate of recorded specs falls below
`--threshold` (0.8). A changed prompt makes a cassette stale; stale specs
count as failures until re-recorded, so prompt changes cannot slip through.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from copilot.agents.base import Catalog
from copilot.agents.pipeline import Pipeline, Run
from copilot.catalog.model import read_snapshot
from copilot.governor import Budget
from copilot.llm import LLM, LLMError
from copilot.replay import RecordingLLM, ReplayLLM

if TYPE_CHECKING:
    from copilot.config import Settings

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.8
BACKOFF = (5.0, 15.0, 45.0)
"""Seconds to wait after each failed attempt at a spec: three attempts, and a pause before
the next spec so it does not land on the same overloaded provider."""


@dataclass
class Outcome:
    id: str
    recorded: bool
    passed: bool = False
    status: str = "unrecorded"
    mismatches: list[str] = field(default_factory=list)
    first_try: bool = False
    corrections: int = 0
    calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)


def load_specs(path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["specs"]
    ids = [s["id"] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("spec ids must be unique")
    return specs


def mismatches(run: Run, expect: dict[str, Any]) -> list[str]:
    if run.status != "reviewed":
        return [f"status {run.status}" + (f" at {run.stage}: " + "; ".join(run.reasons))[:300]]
    assert run.plan is not None
    found: list[str] = []
    actual: dict[str, Any] = {
        "strategy": str(run.plan.strategy),
        "layer": str(run.plan.layer),
        "sources": sorted(run.plan.sources),
        "keys": sorted(run.plan.keys),
    }
    for name, wanted in expect.items():
        want = sorted(wanted) if isinstance(wanted, list) else wanted
        if actual[name] != want:
            found.append(f"{name}: expected {want}, got {actual[name]}")
    return found


def evaluate(spec: dict[str, Any], llm: LLM, catalog: Catalog, cassette: Path) -> Outcome:
    """One spec through plan, generate and review (no sandbox: the same path live and replayed)."""
    outcome = Outcome(spec["id"], recorded=True)
    try:
        run = Pipeline(llm, catalog, budget=Budget()).run(spec["spec"])
    except LLMError as exc:
        outcome.status, outcome.mismatches = "stale", [str(exc)[:300]]
        return outcome
    outcome.status = run.status
    outcome.mismatches = mismatches(run, spec["expect"])
    outcome.passed = not outcome.mismatches
    outcome.calls, outcome.tokens = len(run.attempts), run.tokens
    outcome.corrections = run.corrections
    clean = all(not a.errors for a in run.attempts)
    outcome.first_try = outcome.passed and clean and not run.corrections
    outcome.attempts = [
        {"stage": a.stage, "number": a.number, "ok": not a.errors, "errors": list(a.errors[:3])}
        for a in run.attempts
    ]
    if cassette.exists():
        lines = cassette.read_text(encoding="utf-8").splitlines()
        outcome.seconds = round(sum(float(json.loads(x).get("seconds", 0)) for x in lines if x), 1)
    return outcome


def replay_all(specs: list[dict[str, Any]], catalog: Catalog, cassettes: Path) -> list[Outcome]:
    outcomes = []
    for spec in specs:
        tape = cassettes / f"{spec['id']}.jsonl"
        if not tape.exists():
            outcomes.append(Outcome(spec["id"], recorded=False))
            continue
        outcomes.append(evaluate(spec, ReplayLLM(tape), catalog, tape))
    return outcomes


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    recorded = [o for o in outcomes if o.recorded]
    passed = [o for o in recorded if o.passed]
    n = len(recorded) or 1
    return {
        "specs": len(outcomes),
        "recorded": len(recorded),
        "passed": len(passed),
        "pass_rate": round(len(passed) / n, 3) if recorded else None,
        "first_try_rate": round(sum(o.first_try for o in recorded) / n, 3) if recorded else None,
        "avg_corrections": round(sum(o.corrections for o in recorded) / n, 2) if recorded else None,
        "avg_tokens": round(sum(o.tokens for o in recorded) / n) if recorded else None,
        "avg_seconds": round(sum(o.seconds for o in recorded) / n, 1) if recorded else None,
    }


def gate(
    outcomes: list[Outcome], baseline: dict[str, Any], threshold: float
) -> tuple[bool, list[str]]:
    """(ok, reasons): regressions against the baseline, and the pass-rate floor."""
    reasons = []
    now = {o.id: o for o in outcomes}
    for spec_id in baseline.get("passed", []):
        outcome = now.get(spec_id)
        if outcome is None or not outcome.passed:
            why = "; ".join(outcome.mismatches) if outcome else "spec removed"
            reasons.append(f"regression: {spec_id} passed in the baseline, now fails - {why}")
    summary = summarize(outcomes)
    rate = summary["pass_rate"]
    if rate is not None and rate < threshold:
        reasons.append(
            f"pass rate {rate:.0%} of {summary['recorded']} recorded is below {threshold:.0%}"
        )
    return not reasons, reasons


def live_llm(settings: Settings) -> LLM:
    """The model `record` uses: the client's own retries off, so the backoff in `record`
    is the only retry layer and each attempt is exactly one HTTP request."""
    from copilot.agents.__main__ import make_llm

    return make_llm(settings.model_copy(update={"llm_max_retries": 0}))


@dataclass
class Recording:
    recorded: int = 0
    pending: list[str] = field(default_factory=list)
    """Specs that kept hitting transient provider errors: no cassette, not a failure."""
    error: str | None = None
    """A non-transient error that stopped the batch."""
    quota: bool = False
    """The batch stopped because the provider's daily quota is used up."""


def record(
    specs: list[dict[str, Any]],
    catalog: Catalog,
    cassettes: Path,
    make: Any,
    limit: int,
    force: bool,
    sleep: Callable[[float], None] | None = None,
    backoff: Sequence[float] = BACKOFF,
) -> Recording:
    """Record up to `limit` specs live. Transient provider errors are retried with backoff,
    then the spec is left pending and the next one is tried; an exhausted daily quota or
    any other error stops."""
    pause = sleep or time.sleep
    result = Recording()
    for spec in specs:
        tape = cassettes / f"{spec['id']}.jsonl"
        if result.recorded >= limit:
            break
        if tape.exists() and not force:
            continue
        for attempt, wait in enumerate(backoff, start=1):
            try:
                Pipeline(RecordingLLM(make(), tape), catalog, budget=Budget()).run(spec["spec"])
                break
            except LLMError as exc:
                tape.unlink(missing_ok=True)  # a half-recorded run would replay as stale
                if not exc.transient:
                    result.error = f"{spec['id']}: {exc}"
                    result.quota = exc.quota
                    return result
                print(
                    f"  {spec['id']}: attempt {attempt}/{len(backoff)} - {exc}; waiting {wait:g}s",
                    file=sys.stderr,
                )
                pause(wait)
        else:
            result.pending.append(spec["id"])
            print(f"  {spec['id']}: pending (transient)", file=sys.stderr)
            continue
        result.recorded += 1
        print(f"  recorded {spec['id']}", file=sys.stderr)
    return result


def render(report: dict[str, Any], images: Path) -> list[Path]:
    """SVG images for the README: the eval table, and one failure-and-recovery trace."""
    from rich.console import Console
    from rich.table import Table

    images.mkdir(parents=True, exist_ok=True)
    s = report["summary"]
    table = Table(title="Agent eval suite (replayed from recorded runs)")
    for column in ("spec", "result", "first try", "corrections", "calls", "tokens"):
        table.add_column(column)
    for o in report["outcomes"]:
        if o["recorded"]:
            table.add_row(
                o["id"],
                "pass" if o["passed"] else f"FAIL ({o['status']})",
                "yes" if o["first_try"] else "no",
                str(o["corrections"]),
                str(o["calls"]),
                str(o["tokens"]),
            )
    console = Console(record=True, width=100, file=io.StringIO())
    console.print(table)
    rate = f"{s['pass_rate']:.0%}" if s["pass_rate"] is not None else "n/a"
    first = f"{s['first_try_rate']:.0%}" if s["first_try_rate"] is not None else "n/a"
    console.print(
        f"{s['passed']}/{s['recorded']} recorded specs pass ({rate}); first try {first}; "
        f"avg {s['avg_corrections']} correction(s), {s['avg_tokens']} tokens, {s['avg_seconds']}s"
    )
    written = [images / "eval-report.svg"]
    written[0].write_text(console.export_svg(title="copilot.evals"), encoding="utf-8")
    recovered = next(
        (o for o in report["outcomes"] if o["passed"] and not all(a["ok"] for a in o["attempts"])),
        None,
    )
    if recovered is not None:
        trace = Console(record=True, width=100, file=io.StringIO())
        trace.print(f"[bold]{recovered['id']}[/bold] - a failure and its recovery")
        for a in recovered["attempts"]:
            mark = "[green]ok[/green]" if a["ok"] else "[red]sent back[/red]"
            trace.print(f"  {a['stage']} #{a['number']}: {mark}")
            for error in a["errors"]:
                trace.print(f"      - {error}")
        trace.print(f"  [green]reviewed[/green] after {recovered['calls']} call(s)")
        written.append(images / "recovery.svg")
        written[1].write_text(trace.export_svg(title="copilot run"), encoding="utf-8")
    return written


def main(argv: list[str] | None = None, make: Any = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record unrecorded specs live (uses the model)")
    rec.add_argument("--limit", type=int, default=6)
    rec.add_argument("--only", help="record just this spec id")
    rec.add_argument("--force", action="store_true", help="re-record existing cassettes")
    sub.add_parser("run", help="replay recorded specs and write evals/report.json")
    gt = sub.add_parser("gate", help="replay, then fail on regressions or a low pass rate")
    gt.add_argument("--threshold", type=float, default=THRESHOLD)
    gt.add_argument("--update-baseline", action="store_true")
    sub.add_parser("render", help="write docs/images/*.svg from evals/report.json")
    args = parser.parse_args(argv)

    evals, images = root / "evals", root / "docs" / "images"
    cassettes, report_path = evals / "cassettes", evals / "report.json"
    specs = load_specs(evals / "specs.json")
    catalog = Catalog(read_snapshot(root / "catalog" / "snapshot.jsonl"))

    if args.command == "record":
        if make is None:  # pragma: no cover - the real model
            from copilot.config import get_settings

            settings = get_settings()

            def make() -> LLM:
                return live_llm(settings)

        chosen = [s for s in specs if args.only in (None, s["id"])]
        if not chosen:
            print(f"error: no spec '{args.only}'", file=sys.stderr)
            return 2
        result = record(chosen, catalog, cassettes, make, args.limit, args.force)
        left = sum(not (cassettes / f"{s['id']}.jsonl").exists() for s in specs)
        print(f"recorded {result.recorded}; {left} of {len(specs)} specs still unrecorded")
        if result.pending:
            print(f"pending (transient): {', '.join(result.pending)}")
        if result.quota:
            print("stopped: daily quota exhausted, retry after reset", file=sys.stderr)
            return 2
        if result.error:
            print(f"stopped: {result.error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "render":
        written = render(json.loads(report_path.read_text(encoding="utf-8")), images)
        print("\n".join(str(p.relative_to(root)) for p in written))
        return 0

    outcomes = replay_all(specs, catalog, cassettes)
    summary = summarize(outcomes)
    report = {"summary": summary, "outcomes": [asdict(o) for o in outcomes]}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for o in outcomes:
        if o.recorded and not o.passed:
            print(f"  FAIL {o.id}: {'; '.join(o.mismatches)}", file=sys.stderr)
    rate = "n/a" if summary["pass_rate"] is None else f"{summary['pass_rate']:.0%}"
    print(
        f"{summary['passed']}/{summary['recorded']} recorded specs pass ({rate}); "
        f"{summary['specs'] - summary['recorded']} unrecorded"
    )
    if args.command == "run":
        return 0

    baseline_path = evals / "baseline.json"
    if args.update_baseline:
        passed = sorted(o.id for o in outcomes if o.passed)
        baseline_path.write_text(json.dumps({"passed": passed}, indent=2) + "\n", encoding="utf-8")
        print(f"baseline: {len(passed)} passing spec(s)")
        return 0
    baseline = json.loads(baseline_path.read_text("utf-8")) if baseline_path.exists() else {}
    ok, reasons = gate(outcomes, baseline, args.threshold)
    for reason in reasons:
        print(f"  {reason}", file=sys.stderr)
    print("gate: pass" if ok else "gate: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

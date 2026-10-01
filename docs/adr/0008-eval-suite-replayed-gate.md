# ADR-008: An eval suite recorded in batches, replayed in CI, gated on regressions

- **Status:** accepted
- **Date:** 2026-10-02

## Context

Unit tests prove each gate works; they do not say how often the agents get a
plan right from plain English. That needs a suite of specs run through the
real model - but the free tiers in use allow about 20 requests a day, a run
takes two to four calls, and a CI job that called the model would be slow,
flaky and cost quota on every push.

## Decision

- **50 specs with expectations** in `evals/specs.json`: natural-language
  requests across the catalog's tables and strategies, each with only what
  the spec pins down (strategy, layer, sources, keys). A spec passes when the
  run ends `reviewed` and its plan matches. Runs go through plan, generate
  and review - no sandbox - so recording and replay take the same path.
- **Recorded in batches, replayed everywhere else.** `python -m copilot.evals
  record --limit N` records unrecorded specs live, one cassette each
  (ADR-007), and stops at the first provider error, keeping what it has, so
  the suite fills up over a few days of free quota. `run` and `gate` replay
  every recorded spec with no model, key or network.
- **The gate fails on either signal:** a spec that passed in
  `evals/baseline.json` now fails (a regression), or the pass rate of
  recorded specs is below 0.8. A prompt change makes cassettes stale; stale
  specs fail until re-recorded, so a prompt change cannot pass unnoticed.
  `gate --update-baseline` is run deliberately, after recording.
- **Metrics:** pass rate, first-try rate (no attempt sent back, no
  correction), average corrections, tokens and recorded wall-clock seconds.
  `render` writes them, and one failure-and-recovery trace, as SVG images
  for the README - generated from the replayed report, so they are
  reproducible rather than hand-edited screenshots.

## Consequences

- The suite measures plan and code-review behaviour; sandbox execution stays
  covered by the Docker tests.
- Replayed results only change when the code changes, which is the point:
  the gate catches regressions in checks, prompts and parsing - not model
  drift, which needs a fresh recording.
- Until all 50 specs are recorded, the rate is over the recorded subset;
  the report says how many are unrecorded.

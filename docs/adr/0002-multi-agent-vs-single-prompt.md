# ADR-002: Specialised agents behind deterministic gates, not one prompt

- **Status:** accepted
- **Date:** 2026-10-01

## Context

The copilot turns an English spec ("load orders incrementally into Silver,
keep history for customers") into a PySpark notebook that has been checked
against the lakehouse's standards, run on sample data in a sandbox, tested,
and opened as a pull request. The question is how much of that is one model
call and how much is a pipeline of specialised calls - "agents" - with code
between them.

Multi-agent systems are fashionable, which is a reason to be suspicious of
choosing one. This ADR argues both sides with the costs stated.

## Option A - one prompt

One call gets the spec, the relevant catalog documents and the standards,
and returns the notebook and its tests.

**For it:**

- **Latency.** One round trip. A pipeline of four agents is at least four
  sequential calls before anything has run.
- **Tokens and cost.** Context is sent once. In a pipeline the catalog
  excerpt, the plan and the code are re-sent at every hop, so input tokens
  grow roughly with the number of agents; a critic that reads the whole
  notebook doubles the output it has to process.
- **No hand-off loss.** Agents talk through intermediate artefacts (a plan,
  a review). Every boundary is a place where detail is dropped or
  misread, and an error in the plan is faithfully implemented downstream.
- **Simpler to debug and evaluate.** One prompt version, one trace, one
  place to look. Fewer moving parts to version (ADR-012 in Flagship 2 shows
  how much discipline one versioned prompt already needs).
- **Strong models do plan internally.** Asking for a plan inside the same
  response captures much of the benefit of a separate planner.

**Against it:** the call has to plan, write, self-review and write tests at
once, with nothing between the model and the output to catch a mistake. A
single response cannot run its own code, so "it compiled in my head" is the
only validation - and that is precisely the failure the plan's sandbox and
validator exist to prevent.

## Option B - specialised agents

Planner (reads the catalog, emits a structured plan), Generator (writes code
against the plan, citing the pattern it followed), Critic (reviews against
the standards), Validator (runs it in the sandbox and returns structured
errors), with a bounded correction loop.

**For it:**

- **Places to put deterministic checks.** The value is not the extra model
  calls; it is that each boundary is a typed artefact a program can check.
  A plan that names a table missing from the catalog is rejected before any
  code exists. Code that calls `collect()` unbounded is rejected by a static
  gate before it runs. A sandbox run produces errors the next step can act on.
- **Feedback from reality.** Only a pipeline can run the code and feed the
  error back. Execution, not review, is what catches wrong column names and
  type errors.
- **Different models per stage.** Planning and critique tolerate a cheap
  model; generation benefits from a strong one (step 77's cost governor).
- **Attributable failures.** The trace says which stage failed - plan,
  code, standards or runtime - which is what the agent eval suite (step 78)
  needs to report and what a human needs when the loop escalates.

**Against it:** everything listed for Option A - more latency, more tokens,
hand-off loss, more prompts to version - plus loops: a critic and a
generator can disagree indefinitely, so every loop needs a hard bound and an
escalation path, and the cost of a run becomes variable.

## Decision

**Option B, with three constraints that keep its costs honest:**

1. **Agents only where a deterministic check sits between them.** Planner →
   plan validated against the catalog; Generator → static analysis (ruff,
   mypy, banned-pattern AST checks); Validator → sandbox execution. A stage
   that would only be "another model reading the last model's output" is not
   added.
2. **Bounded everything.** A maximum number of correction rounds, a per-run
   token budget and a hard stop (steps 71, 77); exhausting them escalates to
   a human with a diagnostic report, never silently retries.
3. **Measured against Option A.** The agent eval suite (step 78) runs the
   same specs through a single-prompt baseline and reports first-try success,
   correction rounds, tokens and wall-clock for both. If the pipeline does
   not beat the baseline on success rate by more than it loses on cost, this
   decision is revisited in a new ADR.

## Consequences

- A run is several model calls: slower and more expensive than one prompt.
  The bet is that executed, checked notebooks are worth it; step 78 tests it.
- Every agent's input and output is a typed, recorded artefact, which also
  makes deterministic replay (step 76) and hermetic CI possible.
- The catalog (ADR-003) becomes shared infrastructure: the Planner and
  Generator retrieve from it, and the Critic's standards come from it.

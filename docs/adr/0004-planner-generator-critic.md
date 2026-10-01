# ADR-004: Planner, Generator and Critic - typed hand-offs, a deterministic Critic

- **Status:** accepted
- **Date:** 2026-10-01

## Context

ADR-002 chose specialised agents on one condition: a deterministic check
sits between every pair. This ADR fixes what the first three stages hand to
each other, what checks each hand-off, which model they use, and why the
Critic is not a model.

## Decision

- **Planner -> `Plan`.** A pydantic model: layer, strategy, sources, target,
  keys, incremental column, ordered steps (each citing the catalog pattern it
  follows) and the standards that apply. `check_plan` rejects any source that
  is not a catalog table, any key or column that is not a column of that
  table, any pattern or standard id that is not in the catalog, and a
  strategy in the wrong layer (SCD2 and dedup in Silver, star schema in Gold).
- **Generator -> `NotebookDraft`, rendered by code.** The model returns
  cells; each cites the catalog ids it follows. Code assembles the notebook
  in **Databricks source format** (`.py`, `# COMMAND ----------` between
  cells): it diffs cleanly in a pull request and parses as Python, so every
  later gate reads it with `ast`. Parameters become widgets in a cell the
  code writes. `check_draft` rejects a cell that does not parse, a citation
  that is not in the catalog, a pattern the plan did not choose, and a plan
  pattern no cell follows.
- **Critic = rules, not a model.** Each rule is an `ast` check tied to the
  catalog standard it enforces, reported at its notebook line: unbounded
  `collect`/`toPandas`/`take`, `select("*")` and `SELECT *`, blind
  `append`, storage paths as literals, credentials as literals. Findings go
  back to the Generator verbatim, for a bounded number of rounds.
- **Bounded.** `AGENT_MAX_ATTEMPTS` (default 2) answers per agent and
  `CRITIC_MAX_ROUNDS` (default 1) revisions; then the run is *rejected* with
  the stage and the reasons, and every attempt is kept in the trace.
- **Model: OpenRouter's free models** (`LLM_MODEL`, default `openrouter/free`)
  through a small OpenAI-compatible client with retries on 429/5xx and a cap
  on output tokens. Tests use `ScriptedLLM`, so CI makes no model calls.

## Alternatives

- **An LLM Critic.** It could judge standards no rule can see (is this the
  right natural key?). But it would be another model reading the last
  model's output - exactly what ADR-002 rules out without a check behind
  it - and its findings vary run to run, which makes the correction loop
  harder to bound and evaluate. Deferred: added only if the agent eval
  (step 78) shows the rules miss real problems, and then with findings that
  must cite a real standard and line.
- **Jupyter `.ipynb`.** JSON makes pull-request diffs noisy and every gate
  would first unpack the cells. **Fabric notebooks** fit one platform but are
  harder to check and test outside it. Databricks source converts to either.

## Consequences

- The Critic overlaps the static-analysis gate (step 72), which adds ruff,
  mypy and more AST bans; the Critic's rules move there or are shared.
- Standards about meaning, not syntax, are checked by execution (steps
  69-71) and by the human reviewing the pull request.
- A run costs at least two model calls (plan, generate) and at most
  `2 x AGENT_MAX_ATTEMPTS x (1 + CRITIC_MAX_ROUNDS)`; the trace reports both.

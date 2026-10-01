# ADR-006: A static gate in the sandbox, tests derived by code, and a pull request that never merges

- **Status:** accepted
- **Date:** 2026-10-01

## Context

ADR-005 runs every notebook on sample data, but a Spark session costs tens of
seconds and some mistakes - an undefined name, a typo'd method, a `subprocess`
call - are visible without running anything. The plan's expectations were
hard-coded in the harness, so a reviewer could not read or re-run them. And
the deliverable promised from the start is a pull request, which means a
GitHub credential - the copilot's first write access to anything outside its
sandbox.

## Decision

- **The static gate runs inside the sandbox, before Spark.** The image (now
  `copilot-sandbox:0.2`) pins `ruff==0.16.9`, `mypy==2.3.1` and
  `pytest==9.1.1`. The harness runs `ruff check --isolated --select F,E9,B,S`
  (`spark`, `dbutils`, `display` declared as builtins) on the notebook, and
  mypy on the notebook behind a prelude that declares what Databricks
  injects, so PySpark's own types catch `df.colect()`. Findings come back
  as `static` errors with their notebook line (prelude lines subtracted);
  any finding skips Spark. A tool that cannot run is a finding: the gate
  fails closed. Generated code is never linted, type-checked or imported on
  the host. The Critic's AST rules stay on the host - they are the
  project's own bans (collect, append, secrets), cheap and model-facing.
- **Tests are derived by code, never by a model.** `generate_tests(plan,
  catalog)` writes a pytest file: the target is not empty, its keys exist
  and are never null, incremental/CDC/dedup/SCD2 targets keep one (current)
  row per key, and each source column's data-contract `Expect:` rule
  (`unique`, `not_null`) becomes a test that fails or only warns by its
  `severity`. The harness runs this same file in-process against the live
  session (fixtures `spark`, `target_table`), so what the Validator checked
  is exactly what ships as `tests/test_<target>.py`.
- **Pull requests with least privilege.** `copilot.pr.GitHubPR` talks to one
  repository (`PR_REPO`) with a fine-grained token scoped to it - Contents
  and Pull requests read/write, nothing else. It creates
  `copilot/<target>-<UTC timestamp>` from the base branch, commits the
  notebook and its tests, and opens the PR with `maintainer_can_modify` off.
  It has no merge method; tests assert every request goes to
  `/repos/<PR_REPO>/...` and none to a merge endpoint. The body carries the
  evidence: plan, cited patterns, review, static, execution and test
  results, correction rounds, model calls and tokens. `--open-pr` checks the
  configuration before any model call and opens a PR only for a `ready`
  run. The demo target is a dedicated playground repository, so even a
  leaked token reaches nothing that matters.

## Consequences

- A static failure costs seconds, not a Spark start, and feeds the same
  correction loop with a line number.
- The image is larger (three pinned tools) and must be rebuilt (`0.2`).
- Data contracts are only as good as the catalog's `Expect:` lines; today
  only `customers.customer_id` carries one.
- A human merges every PR. Write access is proven by the first PR, since a
  fine-grained token's permissions cannot be read back through the API.

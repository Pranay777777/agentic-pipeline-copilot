# ADR-005: A Docker sandbox, a Validator that runs every notebook, and two correction rounds

- **Status:** accepted
- **Date:** 2026-10-01

## Context

ADR-002's strongest argument for a pipeline was feedback from reality: only
running the code catches wrong column names, type errors and logic that does
not meet the plan. Running model-written code is also the riskiest thing the
copilot does. Step 69 is a hard gate: isolated container, no network,
read-only mounts, CPU and memory caps, a hard timeout - never `exec()` in
the copilot's own process.

## Decision

- **Sandbox = a pinned image, started with every isolation flag.**
  `sandbox/Dockerfile` is `python:3.12-slim-bookworm` + OpenJDK 17 +
  PySpark 4.0.4 + Delta 4.0.1, with Delta's jars resolved at build time so
  runs need no network. `copilot.sandbox.runner` starts it with
  `--network none`, `--read-only`, `--cap-drop ALL`,
  `--security-opt no-new-privileges`, a non-root user, `--cpus`,
  `--memory` (= swap), `--pids-limit`, a size-capped `/tmp`, the job mounted
  read-only and only `/out` writable. No host environment is passed in. A
  run that exceeds the timeout is killed with `docker kill`.
- **Isolation is proved, not asserted.** `python -m copilot.sandbox check`
  runs a probe inside the real container that tries the network, DNS,
  writing to the job, writing to the root filesystem and reading secrets
  from the environment, and fails unless each is blocked. CI runs it on
  every push, then runs real notebooks in the sandbox.
- **Validator = sample data + expectations from the plan.** Rows are
  generated from the catalog's column types: every plan key appears twice
  with a newer `updated_at` the second time, `*_id` values line up across
  tables, non-key columns are sometimes null. The harness loads them as
  Delta tables, runs the notebook's cells in order with `spark` and a
  `dbutils` stand-in (widgets from the parameter contract, secrets refused),
  then checks the target: not empty, keys present and not null, and one row
  per key for incremental, CDC, dedup and SCD2 (current rows) strategies.
- **Structured errors.** A failure is reported as stage, cell, title,
  *notebook* line, exception type and the exception's first line (Spark's
  query plan dropped) - e.g. `cell 2 'Read new Bronze rows' (notebook line
  28): AnalysisException: ... order_ts cannot be resolved. Did you mean ...`.
- **Parameter contract.** A notebook may declare only `<source>_table`,
  `target_table`, `watermark` and `key_columns`; the Generator is told so and
  `check_draft` enforces it, which is what lets the Validator supply values.
- **Self-correction: one budget, two rounds.** Review findings and execution
  errors both go back to the Generator; `CORRECTION_MAX_ROUNDS=2` revisions
  in total (replacing ADR-004's `CRITIC_MAX_ROUNDS`). If problems remain,
  the run is **escalated**: a diagnostic report (`runs/<target>.report.md`)
  lists every model answer, review and execution with its errors and
  checks, the plan and the last notebook. Runs without a sandbox can be at
  best **reviewed**, never ready.

## Alternatives

- **The official `apache/spark` image** - larger, its Python and Delta
  versions are not ours to pin, and Delta still needs jars at run time.
- **The lakehouse's seed generator for sample data** - realistic volumes,
  but it ties the copilot to the lakehouse's code and dependencies, which
  ADR-003 deliberately avoided; and small generated rows are enough to catch
  wrong columns and broken dedup.
- **gVisor / Firecracker** - stronger isolation, but not available on the
  developer's Windows machine; the flags above are the baseline, a hardened
  runtime can be added with `--runtime` later.

## Consequences

- A run needs Docker and a ~1 GB image; the first build takes minutes.
- Execution adds roughly 20-40 s per round (Spark start-up dominates).
- The checks prove shape (keys, emptiness, duplicates), not business
  correctness; generated pytest assertions (step 73) extend them.

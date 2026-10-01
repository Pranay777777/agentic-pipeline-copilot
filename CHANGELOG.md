# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffold from repo-template.
- ADR-002: specialised agents behind deterministic gates, measured against a single prompt.
- Catalog of metadata-driven-lakehouse v1.0.0 - tables, patterns, standards - with BM25 search and a retrieval benchmark (ADR-003).
- Planner agent: English spec to a structured plan, checked against the catalog (ADR-004).
- Generator agent: Databricks source notebooks whose cells cite the catalog patterns they follow.
- Critic: AST rules for the notebook standards, findings fed back to the Generator, all bounded.
- OpenRouter client with retries and an output-token cap; `ScriptedLLM` for hermetic tests.
- Sandbox: pinned PySpark 4.0.4 + Delta 4.0.1 image, run with no network, read-only filesystem, no capabilities, non-root, CPU/memory/process caps and a hard timeout; `copilot.sandbox check` proves the isolation (ADR-005).
- Validator: runs each notebook on sample data generated from the catalog and checks the target against the plan (non-empty, keys not null, one row per key); errors come back with cell, notebook line and exception.
- Self-correction: review findings and execution errors share a two-round budget; runs that still fail escalate with a diagnostic report of the whole trajectory.
- Static gate: ruff and mypy run on every notebook inside the sandbox (image `copilot-sandbox:0.2`) before Spark starts; findings come back with their notebook line (ADR-006).
- Generated tests: a pytest file derived from the plan and the catalog's data contracts (`unique`, `not_null`, warn or fail by severity) is run by the Validator and ships beside the notebook.
- Pull-request tool: `--open-pr` opens a PR with the notebook and its tests on one repository through a fine-grained token, with the run's evidence in the body; it never merges. `python -m copilot.pr check` verifies the token.
- MCP server (`python -m copilot.mcp_server`): catalog search and lookup, the Critic, generated tests and sandboxed validation as tools; no model calls, no writes, no network (ADR-007).
- Deterministic replay: `--record` writes a run's model calls to a JSONL cassette keyed by request hash; `--replay` answers from it with no model; a recorded real run is replayed in CI.
- Cost governor: every run is capped at `RUN_MAX_CALLS` model calls and `RUN_MAX_TOKENS` tokens and stops hard (`rejected` at stage `budget`).
- Planner check: the target may not overwrite a catalog table.

### Fixed
- Sample data dropped catalog columns that carry a description (e.g. `customers.customer_id`).

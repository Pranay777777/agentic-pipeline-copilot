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

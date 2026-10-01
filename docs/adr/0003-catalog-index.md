# ADR-003: A catalog extracted statically from the lakehouse, searched with BM25

- **Status:** accepted
- **Date:** 2026-10-01

## Context

The agents must write code that fits Flagship 1 (metadata-driven-lakehouse):
its real tables and columns, its patterns (Silver dedup, SCD2 MERGE, a Gold
star schema with unknown members) and its standards. A model's general
knowledge of PySpark does not know any of that.

## Decision

- **Extracted, not hand-written.** `python -m copilot.catalog snapshot`
  reads a lakehouse checkout and emits citable documents:
  - *tables* - columns and types parsed from the `pa.table({...})` literals
    in the seed generator, the function docstring, plus the data contract's
    descriptions, expectations, owner and freshness SLA where one exists;
  - *patterns* - the module docstrings of the transformation code and the
    Context and Decision sections of every ADR;
  - *standards* - CONTRIBUTING, the lint and type-check configuration (parsed
    from TOML), and this repo's `catalog/standards.md`, the rules generated
    notebooks are held to.
- **Read statically, never imported.** Extraction parses source with `ast`,
  YAML and TOML; it never imports or executes lakehouse code, so the copilot
  needs none of the lakehouse's dependencies and extraction cannot run code.
- **Pinned and committed.** Every document records the lakehouse commit it
  came from (`c67e146`, v1.0.0). The snapshot is committed, so search, the
  agents and CI never need the lakehouse checkout.
- **BM25 first.** Catalog questions are full of exact identifiers
  (`customer_id`, `MERGE`, `dq_rule`); the tokenizer keeps snake_case and
  dotted identifiers whole and split. Dense retrieval is added only if the
  benchmark shows misses it would fix.

## Measured

24 labelled planner questions over 49 documents
([report](../results/catalog-retrieval.md)): **Recall@1 0.60, Recall@3 0.85,
Recall@5 0.94, MRR 0.84.** Several questions have two relevant documents,
which caps Recall@1 below 1.0. The two misses at 5 lose to a closely related
document (the Silver standard outranks the Silver module; the quality-engine
documents outrank the quarantine ADR) - candidates for dense fusion or
reranking later, and the labels were not edited after seeing them.

## Consequences

- A lakehouse change needs a re-snapshot to reach the agents; the commit in
  every document makes staleness visible.
- The extractor depends on the lakehouse's conventions (seed generator shape,
  ADR layout); a convention change breaks extraction loudly in the tests
  rather than silently.

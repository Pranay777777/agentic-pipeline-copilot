# Standards for generated PySpark notebooks

Every notebook the copilot generates is held to these rules. The Critic
agent reviews against them and the static-analysis gate enforces the ones a
machine can check. Each rule says why, because a rule without a reason gets
"optimised" away.

## Configuration comes from metadata, never literals

Table names, paths, keys and load types are read from the control plane
(`source_object`, `column_metadata`) or passed as notebook parameters. A
hardcoded path or table name works once and breaks on the next environment;
the lakehouse onboards a source with one metadata row, and a generated
notebook must not undo that.

## No unbounded collect or toPandas

`collect()`, `toPandas()` and `take()` without a small explicit limit pull a
whole distributed dataset onto the driver and fail at production volume.
Aggregate in Spark; if rows must reach the driver, bound them with
`limit(n)` first and say why.

## No credentials in code

Secrets are referenced by name and resolved at run time (the lakehouse's
`credentials` module, ADR-0018). A literal key, password, token or
connection string in a notebook is a leak the moment it is committed.

## Writes are idempotent

Re-running a notebook must not duplicate data. Use Delta `MERGE` on the
natural key for incremental loads, or overwrite a whole partition - never a
blind `append` for data that may be reprocessed.

## Silver keeps one current row per key

Deduplicate on the natural key by the incremental column, newest wins, as
the lakehouse's Silver does; objects whose history matters use SCD2 with
`valid_from`, `valid_to` and an `is_current` flag, never in-place updates.

## Gold joins on surrogate keys with an unknown member

Facts join dimensions on derived surrogate keys; a fact whose dimension row
has not arrived points at the unknown member (key 0) rather than being
dropped, so totals stay correct and the gap is countable.

## Quality rules run before writes

Apply the object's `dq_rule` expectations before writing Silver; failing
rows go to quarantine with the rule that failed, never silently filtered.

## Every notebook ships with tests

A generated notebook comes with pytest assertions on its output (row counts,
key uniqueness, null rules) runnable on sample data in the sandbox. Code
that has not run is not delivered.

## Explicit schemas and column lists

Select columns by name and cast to the declared types; never `select("*")`
into a write. Schema drift is detected and handled (ADR-0007), not passed
through by accident.

# ADR-007: A least-privilege MCP tool layer, cassette replay, and a per-run budget

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Three gaps were left after ADR-006. Other agents and IDEs speak MCP, and the
copilot's catalog and gates are useful to them - but an MCP server is a new
entry point, and the OWASP MCP Top 10 is mostly about tools that do more than
their callers expect. Every CI run of the agents used scripted answers, never
a real model's, and a live run cost tokens and depended on a free tier's
mood. And nothing bounded what one run could spend: each completion had a
token cap, the correction loop had a round cap, but a run as a whole did not.

## Decision

- **MCP server = lookup and gates, never a model or a write.**
  `python -m copilot.mcp_server` (official `mcp` SDK, stdio) exposes
  `search_catalog`, `get_catalog_doc`, `review_notebook`,
  `generate_notebook_tests` (read-only) and `validate_notebook`, which runs
  code only in the Docker sandbox behind the static gate (ADR-005/006).
  No tool calls a model, writes a file, opens a pull request or reaches the
  network; tool annotations say so (`readOnlyHint`, `openWorldHint: false`).
  Inputs are capped (query 500 characters, notebook 200 000, 20 results);
  plans are validated against the catalog before use. Deliberate refusals
  return their reason; any other exception reaches the client only as
  "Error executing tool" - its text stays on the server. Catalog text is
  labelled as data, not instructions. Planning and generation stay behind the
  CLI, where the budget and the human-reviewed PR apply.
- **Replay = one JSONL cassette per run, keyed by the request's hash.**
  `--record PATH` appends every call (request SHA-256, abridged prompt,
  answer, token counts); `--replay PATH` answers only requests whose hash was
  recorded, in recorded order, with no key and no network. A changed prompt,
  catalog or correction message changes the hash and replay fails loudly,
  rather than answering a different question. A real run's cassette is
  committed under `tests/cassettes/`, so CI replays a real model's answers
  through the real planner, generator and critic for free; when a prompt
  changes on purpose, the cassette is re-recorded in the same commit.
- **Cost governor = a per-run budget with a hard stop.** Every call goes
  through `GovernedLLM`: it refuses once the run has made `RUN_MAX_CALLS`
  calls (8 - the worst case of the bounded loop) or spent `RUN_MAX_TOKENS`
  (60 000), and if a call's reported tokens cross the line the run stops at
  once and that answer is discarded. The run ends `rejected` at stage
  `budget`; its trace and diagnostic report carry the usage.

## Consequences

- An MCP client can explore the catalog and check its own notebooks, but
  cannot spend the copilot's tokens or touch GitHub through it.
- Replay is exact, so it is brittle by design: any prompt change needs a new
  recording (one live run).
- The budget counts tokens as the provider reports them; a provider that
  reports none is bounded by calls alone. Money is deliberately not
  estimated: free tiers make price tables misleading.

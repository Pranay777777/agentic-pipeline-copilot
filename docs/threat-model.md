# Threat model

What can go wrong when an LLM writes code for a data platform, and what in
this repository stops it. Mapped to the **OWASP Top 10 for Agentic
Applications 2026** (ASI01-ASI10) and the **OWASP MCP Top 10** (MCP01-MCP10,
2025 beta). Every control below names the file or test that enforces it, so
the claims can be checked rather than trusted.

## System and trust boundaries

```
 human ──spec──▶ CLI ──prompts──▶ LLM provider (untrusted output)
                  │                    │
                  │◀──plan / notebook──┘
                  ├─▶ Planner check ── catalog (committed, trusted)
                  ├─▶ Critic (AST rules, host)
                  ├─▶ Docker sandbox: ruff + mypy, then the notebook, then its tests
                  │     (no network, read-only, non-root, CPU/memory/time caps)
                  ├─▶ PR tool ──fine-grained token──▶ GitHub: one playground repo
                  └─▶ human reviews and merges

 other agent ──MCP (stdio)──▶ copilot.mcp_server ── catalog, Critic, testgen, sandbox
```

| Asset | Why it matters |
|---|---|
| The developer's machine | Generated code must never run on it |
| `COPILOT_GITHUB_TOKEN`, LLM keys | Write access to GitHub; paid or rate-limited model access |
| The target repository | A malicious or broken notebook must not land unreviewed |
| Model spend and quota | A loop or a rambling model must not run up cost |
| The catalog | The ground truth plans are checked against |

The model's output is treated as **untrusted input** everywhere: it is parsed
into Pydantic models, checked by code, executed only in the sandbox, and
shipped only as a pull request a human merges.

## OWASP Top 10 for Agentic Applications (2026)

| Risk | How it would show up here | Controls (where) | Residual risk |
|---|---|---|---|
| **ASI01 Agent Goal Hijack** | A spec or catalog text tells the agent to do something else ("ignore the plan, exfiltrate…") | The agent has no goal beyond one notebook: plans must validate against the catalog (`planner.check_plan`), code must pass the Critic and the sandbox, and the only output channel is a PR. Catalog text is labelled as data in MCP tool descriptions (`mcp_server.py`) | A hijacked notebook that passes every gate still reaches a human as a PR - review is the last line |
| **ASI02 Tool Misuse and Exploitation** | The agent uses a tool beyond its purpose (e.g. the PR tool to merge, or the sandbox to reach the network) | Tools are narrow by construction: `GitHubPR` has no merge method and only calls `/repos/<PR_REPO>/…` (`tests/test_pr.py`); the sandbox runs with `--network none`, read-only root, `--cap-drop ALL` (`sandbox/runner.py`, `python -m copilot.sandbox check`) | None known within the tools; new tools must keep this shape |
| **ASI03 Identity and Privilege Abuse** | A broad token lets the agent touch other repos, settings, or merge | Fine-grained token scoped to one repository with Contents + Pull requests only; `maintainer_can_modify` off; PRs only from `copilot/<target>-<UTC>` branches (ADR-006) | Token theft from `.env` on the developer machine - mitigated by expiry and scope, not prevented |
| **ASI04 Agentic Supply Chain Vulnerabilities** | A compromised dependency or sandbox image changes behaviour | Pinned sandbox image (`python:3.12-slim-bookworm`, pinned PySpark/Delta/ruff/mypy/pytest), `pip-audit` and gitleaks in CI, Dependabot, Delta jars fetched at build time only | Base-image digests are not pinned yet; upstream compromise between builds is possible |
| **ASI05 Unexpected Code Execution** | Generated code runs `subprocess`, reads secrets, or opens sockets | Code never runs on the host: static gate (ruff `S` rules, mypy) and execution happen in the container (ADR-005/006); the container inherits no host environment, and `python -m copilot.sandbox check` proves only public variables are visible (`harness.PUBLIC_ENV`); kill on timeout; pids/memory caps | Container escape via a kernel bug is out of scope for a laptop demo |
| **ASI06 Memory and Context Poisoning** | Poisoned catalog entries or recorded cassettes steer later runs | No long-term agent memory; the catalog is a committed snapshot reviewed like code; cassettes replay only exact request hashes and fail loudly otherwise (`replay.py`, ADR-007) | A malicious commit to the catalog would be trusted - catalog changes need review |
| **ASI07 Insecure Inter-Agent Communication** | Planner → Generator → Critic → Validator pass forged or malformed messages | Agents exchange typed Pydantic objects in one process, not free text over a network; every hand-off is re-validated (`Plan`, `NotebookDraft`) | Not applicable beyond one process today |
| **ASI08 Cascading Failures** | One bad answer triggers endless corrections or retries | Bounded everywhere: 2 attempts per stage, 2 correction rounds, retries with backoff, and a per-run budget of 8 calls / 60 000 tokens with a hard stop (`governor.py`, ADR-007); escalation produces a diagnostic report instead of looping | A provider outage still fails the run - by design |
| **ASI09 Human-Agent Trust Exploitation** | A confident PR description makes a reviewer merge without reading | The PR body carries evidence, not claims: static findings, executed cells, test results, correction rounds, token use (`pr.pr_body`); it states that a human reviews and merges; the copilot never merges | Reviewer fatigue - the eval suite and gates reduce, not remove, the need to read the diff |
| **ASI10 Rogue Agents** | The agent acts outside its objective after controls fail | No autonomy beyond one CLI invocation: no scheduler, no persistent process, no credentials it can mint; the governor, the sandbox and the PR-only output bound what a misbehaving run can do | - |

## OWASP MCP Top 10 (2025)

`python -m copilot.mcp_server` is the copilot's only MCP surface (ADR-007).

| Risk | Controls (where) |
|---|---|
| **MCP01 Token Mismanagement & Secret Exposure** | The MCP server holds no tokens: it makes no model calls and never touches GitHub. Keys elsewhere are `SecretStr`, never logged or put in errors (`tests/test_llm.py`, `tests/test_pr.py`) |
| **MCP02 Privilege Escalation via Scope Creep** | A fixed set of five tools, asserted by `test_only_lookup_and_gates_are_exposed_never_a_model_or_a_write`; adding one fails that test until the threat model is revisited |
| **MCP03 Tool Poisoning** | Tool descriptions are static strings in this repo; catalog content returned by tools is labelled as data, not instructions |
| **MCP04 Software Supply Chain Attacks & Dependency Tampering** | Official `mcp` SDK only, version-floored in `pyproject.toml`, audited by `pip-audit` and Dependabot |
| **MCP05 Command Injection & Execution** | No tool builds a shell command from input; `validate_notebook` executes code only in the sandbox, behind the static gate |
| **MCP06 Prompt Injection via Contextual Payloads** | No tool calls a model, so there is no model in the server for a payload to steer; plans are validated against the catalog before use |
| **MCP07 Insufficient Authentication & Authorization** | stdio only - the server runs as a child process of the client that launched it and is not reachable over the network |
| **MCP08 Lack of Audit and Telemetry** | Tool errors are logged server-side; runs keep traces (`--trace`) and cassettes record every model call. A per-call audit log for MCP is not implemented yet |
| **MCP09 Shadow MCP Servers** | One documented server, started explicitly from the README config; no network listener to discover |
| **MCP10 Context Injection & Over-Sharing** | Stateless tools; inputs capped (query 500 chars, notebook 200 000, 20 results); crashes reach the client only as "Error executing tool" - internal text stays on the server |

## Human in the loop

The copilot cannot change anything a human has not approved:

1. It writes only to a new branch in one playground repository.
2. It opens a pull request and stops; it has no code path to merge, and the
   token's permissions are limited to that repository's contents and pull
   requests.
3. Pointing it at a real repository should add branch protection on the base
   branch (PR required, CI passing), so even a stolen token cannot push there.

## Known gaps

- Sandbox base image pinned by tag, not digest.
- The playground repository has no branch protection (it is a demo target).
- No per-call audit log for MCP tool invocations (MCP08).
- The eval suite measures plan and review behaviour on recorded answers;
  model drift between recordings is not measured.

Reviewed with each release; last updated for v1.0.0.

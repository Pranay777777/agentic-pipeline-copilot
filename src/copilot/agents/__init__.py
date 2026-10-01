"""The agents (ADR-002, ADR-004): Planner, Generator and Critic.

Each agent's output is a typed artefact that a deterministic check reads
before the next stage starts. A model answer the check rejects is sent back
with the reasons, a bounded number of times; then the stage is rejected and
the run stops, saying which stage failed and why.
"""

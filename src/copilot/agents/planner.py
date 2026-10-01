"""Planner (step 66): an English spec in, a structured plan out.

The plan is the first typed boundary of ADR-002. Before any code exists, a
program checks it against the catalog: every source is a real table, every
key and column is a real column of that table, every pattern and standard is
a real catalog document, and the strategy fits the layer. A plan that names
something the lakehouse does not have is sent back with the reasons, and
rejected if the model cannot fix it within its budget.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from copilot.agents.base import Catalog, Exchange, ask_json
from copilot.catalog.index import CatalogIndex
from copilot.catalog.model import Kind
from copilot.llm import LLM

IDENT = r"^[a-z][a-z0-9_]*$"


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class Strategy(StrEnum):
    FULL_RELOAD = "full_reload"
    INCREMENTAL = "incremental"
    CDC_MERGE = "cdc_merge"
    DEDUP_LATEST = "dedup_latest"
    SCD2 = "scd2"
    STAR_SCHEMA = "star_schema"


NEEDS_INCREMENTAL_COLUMN = {Strategy.INCREMENTAL, Strategy.CDC_MERGE, Strategy.DEDUP_LATEST}
LAYER_OF = {
    Strategy.DEDUP_LATEST: Layer.SILVER,
    Strategy.SCD2: Layer.SILVER,
    Strategy.STAR_SCHEMA: Layer.GOLD,
}


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str = Field(min_length=3, max_length=300)
    table: str | None = Field(
        default=None,
        description="A source table this step reads, or the plan's target for the step "
        "that writes it; null if the step works on an earlier step's result.",
    )
    columns: list[str] = Field(
        default_factory=list, description="Source columns the step uses, exactly as listed."
    )
    pattern: str = Field(
        description="The catalog pattern id this step follows, "
        "e.g. pattern:module.transform-silver."
    )


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str = Field(min_length=3, max_length=300)
    layer: Layer
    strategy: Strategy
    sources: list[str] = Field(min_length=1, max_length=6)
    target: str = Field(pattern=IDENT, max_length=64, description="The table this notebook writes.")
    keys: list[str] = Field(min_length=1)
    incremental_column: str | None = None
    steps: list[PlanStep] = Field(min_length=1, max_length=12)
    standards: list[str] = Field(default_factory=list)

    @property
    def patterns(self) -> list[str]:
        return list(dict.fromkeys(s.pattern for s in self.steps))


def check_plan(plan: Plan, catalog: Catalog) -> list[str]:
    """Everything wrong with the plan, judged against the catalog. Empty means valid."""
    errors: list[str] = []
    tables = catalog.columns
    for source in plan.sources:
        if source not in tables:
            errors.append(f"source '{source}' is not a catalog table ({', '.join(sorted(tables))})")
    known = frozenset().union(*(tables.get(s, frozenset()) for s in plan.sources))
    if plan.target in plan.sources or plan.target in tables:
        errors.append(
            f"target '{plan.target}' would overwrite a catalog table - name it for its layer, "
            f"e.g. '{plan.sources[0]}_{plan.layer}'"
        )
    for key in plan.keys:
        if key not in known:
            errors.append(f"key '{key}' is not a column of {', '.join(plan.sources)}")
    if plan.incremental_column is not None and plan.incremental_column not in known:
        errors.append(f"incremental_column '{plan.incremental_column}' is not a source column")
    if plan.strategy in NEEDS_INCREMENTAL_COLUMN and plan.incremental_column is None:
        errors.append(f"strategy '{plan.strategy}' needs an incremental_column")
    expected = LAYER_OF.get(plan.strategy)
    if expected is not None and plan.layer is not expected:
        errors.append(f"strategy '{plan.strategy}' belongs in {expected}, not {plan.layer}")
    for n, step in enumerate(plan.steps, 1):
        # The target is built from the sources, so a step on it uses source columns.
        on_target = step.table == plan.target
        if step.table is not None and step.table not in plan.sources and not on_target:
            errors.append(
                f"step {n}: table '{step.table}' is neither a source "
                f"({', '.join(plan.sources)}) nor the target ('{plan.target}')"
            )
        columns = tables.get(step.table, frozenset()) if step.table and not on_target else known
        errors += [
            f"step {n}: column '{c}' is not in {step.table or 'the sources'}"
            for c in step.columns
            if c not in columns
        ]
        if not catalog.is_a(step.pattern, Kind.PATTERN):
            errors.append(f"step {n}: '{step.pattern}' is not a catalog pattern")
    errors += [
        f"'{s}' is not a catalog standard"
        for s in plan.standards
        if not catalog.is_a(s, Kind.STANDARD)
    ]
    return errors


SYSTEM = """You are the Planner of a pipeline copilot for a metadata-driven lakehouse.
Turn the user's spec into a plan for ONE PySpark notebook. Use only the tables,
columns, patterns and standards listed in the catalog excerpt - a program checks
every name against the catalog and rejects anything that is not there.

Rules:
- sources: catalog table names (without the "table:" prefix); target: the table written.
- a step's table is a source it reads, or the target for the step that writes it.
- keys, incremental_column, step columns: columns of those tables, exactly as listed.
- every step cites the catalog pattern id it follows (an id starting "pattern:").
- standards: ids of the catalog standards the notebook must respect ("standard:...").
- strategies: full_reload, incremental, cdc_merge, dedup_latest (silver), scd2 (silver),
  star_schema (gold). incremental, cdc_merge and dedup_latest need incremental_column.

Reply with one JSON object matching this JSON Schema, and nothing else:
"""


def _context(spec: str, catalog: Catalog, index: CatalogIndex) -> str:
    tables = [d for d in catalog.docs.values() if d.kind is Kind.TABLE]
    patterns = [h.doc for h in index.search(spec, k=6, kind=Kind.PATTERN)]
    standards = [catalog.docs[i] for i in catalog.ids(Kind.STANDARD)]
    parts = ["## Tables"] + [f"### {d.id}\n{d.text}" for d in tables]
    parts += ["## Patterns most relevant to the spec"]
    parts += [f"### {d.id} - {d.title}\n{d.text[:1200]}" for d in patterns]
    parts += ["## Standards"] + [f"- {d.id}: {d.title}" for d in standards]
    return "\n\n".join(parts)


class Planner:
    def __init__(self, llm: LLM, catalog: Catalog, index: CatalogIndex, max_attempts: int) -> None:
        self.llm, self.catalog, self.index = llm, catalog, index
        self.max_attempts = max_attempts

    def plan(self, spec: str, exchange: Exchange | None = None) -> Plan:
        schema = json.dumps(Plan.model_json_schema())
        messages = [
            {"role": "system", "content": SYSTEM + schema},
            {
                "role": "user",
                "content": f"# Catalog excerpt\n\n{_context(spec, self.catalog, self.index)}"
                f"\n\n# Spec\n\n{spec}",
            },
        ]
        return ask_json(
            self.llm,
            "plan",
            messages,
            Plan,
            lambda p: check_plan(p, self.catalog),
            self.max_attempts,
            exchange,
        )

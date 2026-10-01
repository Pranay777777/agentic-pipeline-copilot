"""Retrieval benchmark for the catalog: labelled questions, recall@k and MRR."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from copilot.catalog.index import CatalogIndex


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    text: str
    relevant: list[str] = Field(min_length=1)


class QuerySet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    queries: list[Query] = Field(min_length=1)


def read_queries(path: Path) -> QuerySet:
    return QuerySet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Result:
    query: Query
    ranked: list[str]

    def recall(self, k: int) -> float:
        return len(set(self.ranked[:k]) & set(self.query.relevant)) / len(self.query.relevant)

    @property
    def reciprocal_rank(self) -> float:
        for position, doc_id in enumerate(self.ranked, start=1):
            if doc_id in self.query.relevant:
                return 1.0 / position
        return 0.0


def run(index: CatalogIndex, queries: QuerySet, depth: int = 10) -> list[Result]:
    return [Result(q, [h.doc.id for h in index.search(q.text, k=depth)]) for q in queries.queries]


def unknown_ids(index: CatalogIndex, queries: QuerySet) -> set[str]:
    known = {d.id for d in index.docs}
    return {i for q in queries.queries for i in q.relevant if i not in known}


def markdown(results: list[Result], docs: int, commit: str) -> str:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines = [
        f"# Catalog retrieval - {date.today().isoformat()}",
        "",
        f"{len(results)} labelled questions over {docs} catalog documents extracted from "
        f"metadata-driven-lakehouse@{commit}; BM25.",
        "",
        "| Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---|---|---|",
        "| "
        + " | ".join(
            f"{value:.2f}"
            for value in (
                mean([r.recall(1) for r in results]),
                mean([r.recall(3) for r in results]),
                mean([r.recall(5) for r in results]),
                mean([r.reciprocal_rank for r in results]),
            )
        )
        + " |",
        "",
        "## Misses at 5",
        "",
    ]
    misses = [r for r in results if r.recall(5) < 1.0]
    lines += [
        f"- {r.query.id}: wanted {', '.join(r.query.relevant)}; got "
        f"{', '.join(r.ranked[:5]) or 'nothing'}"
        for r in misses
    ] or ["None."]
    return "\n".join(lines) + "\n"

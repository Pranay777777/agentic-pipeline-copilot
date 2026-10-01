"""One catalog document: something an agent can cite."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Kind(StrEnum):
    TABLE = "table"
    PATTERN = "pattern"
    STANDARD = "standard"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    repo: str
    commit: str
    path: str


class CatalogDoc(BaseModel):
    """A citable unit. IDs are stable across snapshots of the same source."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(pattern=r"^(table|pattern|standard):[a-z0-9_.#-]+$")
    kind: Kind
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: Source
    tags: list[str] = Field(default_factory=list)


def read_snapshot(path: Path) -> list[CatalogDoc]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [CatalogDoc.model_validate_json(line) for line in lines if line.strip()]


def write_snapshot(path: Path, docs: list[CatalogDoc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(d.model_dump_json() + "\n" for d in sorted(docs, key=lambda d: d.id))
    path.write_text(body, encoding="utf-8", newline="\n")

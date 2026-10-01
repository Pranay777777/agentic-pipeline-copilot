"""Sample rows for a notebook's sources, generated from the catalog's column types.

Deterministic and shaped to exercise the patterns: every plan key appears
twice with a newer `updated_at` the second time (so deduplication and MERGE
have something to do), `*_id` columns share values across tables (so joins
match), timestamps are epoch integers like the lakehouse's Bronze, and
non-key columns are sometimes null.
"""

from __future__ import annotations

import ast
import re
from typing import Any

# "- customer_id (string)" optionally followed by " - description. Expect: {...}."
_COLUMN = re.compile(r"^- ([a-z_][a-z0-9_]*) \(([a-z0-9]+)\)(?: - (.*))?$", re.MULTILINE)
_EXPECT = re.compile(r"Expect: (\{.*\})")
EPOCH = 1_767_225_600  # 2026-01-01T00:00:00Z


def columns(table_text: str) -> list[tuple[str, str]]:
    """(name, type) pairs from a catalog table document."""
    return [(name, kind) for name, kind, _ in _COLUMN.findall(table_text)]


def contract(table_text: str) -> dict[str, dict[str, Any]]:
    """Each column's data-contract expectations, e.g. {"customer_id": {"unique": True, ...}}."""
    found: dict[str, dict[str, Any]] = {}
    for name, _, description in _COLUMN.findall(table_text):
        match = _EXPECT.search(description or "")
        if match:
            value = ast.literal_eval(match.group(1))
            if isinstance(value, dict):
                found[name] = value
    return found


def rows(cols: list[tuple[str, str]], keys: list[str], n: int = 24) -> list[dict[str, Any]]:
    distinct = max(n // 2, 1)
    out = []
    for i in range(n):
        row: dict[str, Any] = {}
        for name, kind in cols:
            row[name] = _value(name, kind, i, keys, distinct)
        out.append(row)
    return out


def _value(name: str, kind: str, i: int, keys: list[str], distinct: int) -> Any:
    if name in keys:
        k = i % distinct
        return f"{name.removesuffix('_id')}-{k:03d}" if kind == "string" else k
    if name == "updated_at":
        return EPOCH + i * 60
    if name.endswith("_id"):
        k = i % 10
        return f"{name.removesuffix('_id')}-{k:03d}" if kind == "string" else k
    if i % 7 == 6:
        return None
    if kind in {"int64", "int32"}:
        return EPOCH + i * 3600 if name.endswith("_at") else i % 5
    if kind == "float64":
        return round(10 + i * 1.5, 2)
    if kind == "bool":
        return i % 2 == 0
    return f"{name}-{i % 4}"

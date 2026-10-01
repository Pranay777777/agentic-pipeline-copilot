"""Extract catalog documents from a checkout of metadata-driven-lakehouse.

Read statically - nothing from the lakehouse is imported or executed - so
the copilot never needs the lakehouse's dependencies, and extraction cannot
run its code:

- **tables**: each `build_<table>` in `src/lakehouse/seed/generator.py`
  returns a `pa.table({...})` literal; its keys are the columns and each
  `type=pa.<type>()` is the column type. The function's docstring describes
  the table. A data contract in `contracts/` adds column descriptions,
  expectations, owner and freshness SLA.
- **patterns**: the module docstrings of the transformation code (Silver,
  SCD2, Gold, ingestion, quality, privacy) and the Decision section of every
  ADR - the lakehouse's own account of how and why it does things.
- **standards**: `CONTRIBUTING.md`, the lint and type-check configuration,
  and this repo's `catalog/standards.md` - the rules generated notebooks are
  held to.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

from copilot.catalog.model import CatalogDoc, Kind, Source

REPO = "Pranay777777/metadata-driven-lakehouse"
GENERATOR = Path("src/lakehouse/seed/generator.py")
PATTERN_MODULES = (
    "transform/silver.py",
    "transform/scd2.py",
    "transform/gold.py",
    "ingest",
    "quality",
    "privacy.py",
    "lineage.py",
    "maintenance.py",
    "replay.py",
    "audit.py",
)
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG.sub("-", text.casefold()).strip("-")[:60]


def commit_of(checkout: Path) -> str:
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(checkout), "rev-parse", "--short=7", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _type_of(value: ast.expr) -> str:
    if isinstance(value, ast.Call):
        for kw in value.keywords:
            if kw.arg == "type" and isinstance(kw.value, ast.Call):
                func = kw.value.func
                if isinstance(func, ast.Attribute):
                    return func.attr
        if isinstance(value.func, ast.Name) and value.func.id == "_ids":
            return "string"
    return "unknown"


def seed_tables(source: str) -> dict[str, dict[str, Any]]:
    """{table: {"doc": docstring, "columns": [(name, type)]}} from generator.py."""
    tables: dict[str, dict[str, Any]] = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("build_")):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "table"
                and call.args
                and isinstance(call.args[0], ast.Dict)
            ):
                columns = [
                    (key.value, _type_of(value))
                    for key, value in zip(call.args[0].keys, call.args[0].values, strict=True)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
                tables[node.name.removeprefix("build_")] = {
                    "doc": ast.get_docstring(node) or "",
                    "columns": columns,
                }
    return tables


def _contract(checkout: Path, table: str) -> dict[str, Any] | None:
    for path in sorted(checkout.glob("contracts/**/*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("object") == table:
            data["_path"] = path.relative_to(checkout).as_posix()
            return data
    return None


def table_docs(checkout: Path, commit: str) -> list[CatalogDoc]:
    docs = []
    tables = seed_tables((checkout / GENERATOR).read_text(encoding="utf-8"))
    for name, info in tables.items():
        lines = [info["doc"].strip(), "", "Columns:"]
        contract = _contract(checkout, name)
        described = {c["name"]: c for c in (contract or {}).get("columns", [])}
        for column, kind in info["columns"]:
            extra = described.get(column, {})
            note = f" - {extra['description'].strip()}" if extra.get("description") else ""
            expect = f" Expect: {extra['expect']}." if extra.get("expect") else ""
            lines.append(f"- {column} ({kind}){note}{expect}")
        path = GENERATOR.as_posix()
        if contract:
            lines += [
                "",
                f"Contract owner {contract.get('owner', '-')}; freshness SLA "
                f"{contract.get('freshness_sla_minutes', '-')} minutes on "
                f"{contract.get('freshness_column', '-')}.",
            ]
            path += f", {contract['_path']}"
        docs.append(
            CatalogDoc(
                id=f"table:{name}",
                kind=Kind.TABLE,
                title=f"Source table {name}",
                text="\n".join(lines).strip(),
                source=Source(repo=REPO, commit=commit, path=path),
                tags=[c for c, _ in info["columns"]],
            )
        )
    return docs


def _module_docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


def _sections(markdown: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = "_intro"
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            continue
        out[current] = out.get(current, "") + line + "\n"
    return out


def pattern_docs(checkout: Path, commit: str) -> list[CatalogDoc]:
    docs = []
    base = checkout / "src/lakehouse"
    for entry in PATTERN_MODULES:
        target = base / entry
        if not target.exists():  # an older or newer lakehouse may not have every module
            continue
        files = [target] if target.suffix == ".py" else sorted(target.glob("*.py"))
        for path in files:
            text = _module_docstring(path)
            if len(text) < 80:
                continue
            rel = path.relative_to(checkout).as_posix()
            name = path.relative_to(base).with_suffix("").as_posix().replace("/", ".")
            docs.append(
                CatalogDoc(
                    id=f"pattern:module.{slug(name)}",
                    kind=Kind.PATTERN,
                    title=f"{name}: {text.splitlines()[0].rstrip('.')}",
                    text=text,
                    source=Source(repo=REPO, commit=commit, path=rel),
                    tags=[name],
                )
            )
    for path in sorted((checkout / "docs/adr").glob("[0-9][0-9][0-9][0-9]-*.md")):
        if path.name.startswith(("0000", "0001")):
            continue
        markdown = path.read_text(encoding="utf-8")
        title = next(
            (ln[2:].strip() for ln in markdown.splitlines() if ln.startswith("# ")), path.stem
        )
        sections = _sections(markdown)
        body = (
            "\n".join(
                sections[k].strip() for k in ("Context", "Decision") if sections.get(k, "").strip()
            )
            or markdown
        )
        docs.append(
            CatalogDoc(
                id=f"pattern:adr.{path.name[:4]}",
                kind=Kind.PATTERN,
                title=title,
                text=body.strip(),
                source=Source(repo=REPO, commit=commit, path=path.relative_to(checkout).as_posix()),
                tags=["adr"],
            )
        )
    return docs


def standard_docs(checkout: Path, commit: str, own_standards: Path) -> list[CatalogDoc]:
    docs = []
    contributing = checkout / "CONTRIBUTING.md"
    if contributing.exists():
        docs.append(
            CatalogDoc(
                id="standard:lakehouse.contributing",
                kind=Kind.STANDARD,
                title="Lakehouse contribution rules",
                text=contributing.read_text(encoding="utf-8"),
                source=Source(repo=REPO, commit=commit, path="CONTRIBUTING.md"),
                tags=["process"],
            )
        )
    pyproject = checkout / "pyproject.toml"
    if pyproject.exists():
        tool = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {})
        lines = ["Code passes ruff and mypy with this configuration:"]
        for name, table in (
            ("ruff", tool.get("ruff", {})),
            ("ruff.lint", tool.get("ruff", {}).get("lint", {})),
            ("mypy", tool.get("mypy", {})),
        ):
            settings = {k: v for k, v in table.items() if not isinstance(v, dict)}
            if settings:
                lines.append(f"[tool.{name}]")
                lines += [f"{key} = {value!r}" for key, value in settings.items()]
        docs.append(
            CatalogDoc(
                id="standard:lakehouse.lint",
                kind=Kind.STANDARD,
                title="Lint and type-check configuration",
                text="\n".join(lines),
                source=Source(repo=REPO, commit=commit, path="pyproject.toml"),
                tags=["lint"],
            )
        )
    sections = _sections(own_standards.read_text(encoding="utf-8"))
    for heading, body in sections.items():
        if heading == "_intro" or not body.strip():
            continue
        docs.append(
            CatalogDoc(
                id=f"standard:notebook.{slug(heading)}",
                kind=Kind.STANDARD,
                title=heading,
                text=body.strip(),
                source=Source(
                    repo="Pranay777777/agentic-pipeline-copilot",
                    commit="HEAD",
                    path=own_standards.as_posix(),
                ),
                tags=["notebook"],
            )
        )
    return docs


def extract(checkout: Path, own_standards: Path) -> list[CatalogDoc]:
    commit = commit_of(checkout)
    return [
        *table_docs(checkout, commit),
        *pattern_docs(checkout, commit),
        *standard_docs(checkout, commit, own_standards),
    ]

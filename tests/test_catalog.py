"""The catalog (step 65): static extraction, BM25 search, benchmark."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from copilot.catalog import __main__ as cli
from copilot.catalog.bench import QuerySet, Result, markdown, read_queries, run, unknown_ids
from copilot.catalog.extract import extract, seed_tables, slug
from copilot.catalog.index import CatalogIndex, tokens
from copilot.catalog.model import CatalogDoc, Kind, Source, read_snapshot, write_snapshot

SNAPSHOT = Path("catalog/snapshot.jsonl")

GENERATOR = '''
import pyarrow as pa

def build_widgets(cfg, rng):
    """One row per widget."""
    return pa.table(
        {
            "widget_id": _ids("w", 3),
            "weight": pa.array([1.0], type=pa.float64()),
            "label": pa.array(["x"]),
        }
    )

def helper():
    return 1
'''


def doc(ident: str, title: str, text: str, kind: Kind = Kind.PATTERN) -> CatalogDoc:
    return CatalogDoc(
        id=ident, kind=kind, title=title, text=text, source=Source(repo="r", commit="c", path="p")
    )


@pytest.fixture
def lakehouse(tmp_path: Path) -> Path:
    """A miniature lakehouse checkout, so extraction is tested without the real repo."""
    root = tmp_path / "lakehouse"
    (root / "src/lakehouse/seed").mkdir(parents=True)
    (root / "src/lakehouse/seed/generator.py").write_text(GENERATOR, encoding="utf-8")
    (root / "src/lakehouse/transform").mkdir()
    (root / "src/lakehouse/transform/silver.py").write_text(
        '"""Silver: conformed, deduplicated data.\n\nNewest row per natural key wins, by the '
        'incremental column, so reruns converge."""\n',
        encoding="utf-8",
    )
    (root / "src/lakehouse/transform/gold.py").write_text('"""Short."""\n', encoding="utf-8")
    (root / "contracts/seed").mkdir(parents=True)
    (root / "contracts/seed/widgets.yml").write_text(
        "object: widgets\nowner: team@example.com\nfreshness_sla_minutes: 60\n"
        "freshness_column: updated_at\ncolumns:\n  - name: widget_id\n"
        "    description: Natural key.\n"
        "    expect: {unique: true}\n",
        encoding="utf-8",
    )
    (root / "docs/adr").mkdir(parents=True)
    (root / "docs/adr/0001-record.md").write_text("# ADR-001: Record\n", encoding="utf-8")
    (root / "docs/adr/0005-scd2.md").write_text(
        "# ADR-005: SCD2 in Silver\n\n## Context\n\nHistory matters.\n\n## Decision\n\n"
        "Keep every version.\n\n## Consequences\n\nMore rows.\n",
        encoding="utf-8",
    )
    (root / "CONTRIBUTING.md").write_text("Open a PR; CI must be green.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nselect = ["E", "F"]\n\n[tool.mypy]\nstrict = true\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "init",
            "--allow-empty",
        ],
        check=True,
    )
    return root


def test_seed_tables_are_read_statically() -> None:
    tables = seed_tables(GENERATOR)
    assert tables == {
        "widgets": {
            "doc": "One row per widget.",
            "columns": [("widget_id", "string"), ("weight", "float64"), ("label", "unknown")],
        }
    }


def test_extraction_covers_tables_patterns_and_standards(lakehouse: Path, tmp_path: Path) -> None:
    standards = tmp_path / "standards.md"
    standards.write_text(
        "# Standards\n\nIntro.\n\n## No collect\n\nBound it.\n\n## Empty\n\n", encoding="utf-8"
    )
    docs = {d.id: d for d in extract(lakehouse, standards)}
    assert set(docs) == {
        "table:widgets",
        "pattern:module.transform-silver",
        "pattern:adr.0005",
        "standard:lakehouse.contributing",
        "standard:lakehouse.lint",
        "standard:notebook.no-collect",
    }
    table = docs["table:widgets"]
    assert "- widget_id (string) - Natural key. Expect: {'unique': True}." in table.text
    assert "Contract owner team@example.com; freshness SLA 60 minutes" in table.text
    assert table.source.path.endswith("contracts/seed/widgets.yml")
    assert len(table.source.commit) == 7
    assert docs["pattern:adr.0005"].text == "History matters.\nKeep every version."
    assert "select = ['E', 'F']" in docs["standard:lakehouse.lint"].text
    assert "strict = True" in docs["standard:lakehouse.lint"].text
    assert docs["standard:notebook.no-collect"].source.repo.endswith("agentic-pipeline-copilot")


def test_snapshots_round_trip_sorted(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write_snapshot(path, [doc("pattern:b", "B", "b"), doc("pattern:a", "A", "a")])
    assert [d.id for d in read_snapshot(path)] == ["pattern:a", "pattern:b"]
    assert b"\r\n" not in path.read_bytes()


def test_tokens_keep_identifiers_whole_and_split() -> None:
    assert tokens("The customer_id and dq.rule") == [
        "customer_id",
        "customer",
        "id",
        "dq.rule",
        "dq",
        "rule",
    ]
    assert slug("Gold joins: on keys!") == "gold-joins-on-keys"


def test_search_ranks_filters_and_breaks_ties() -> None:
    index = CatalogIndex(
        [
            doc("pattern:scd2", "SCD2 history", "Keep every version with valid_from."),
            doc("pattern:silver", "Silver dedup", "Newest row per key wins."),
            doc("standard:history", "History standard", "Keep history with SCD2.", Kind.STANDARD),
        ]
    )
    hits = index.search("keep history scd2")
    assert hits[0].doc.id in {"pattern:scd2", "standard:history"} and len(hits) == 2
    assert [h.doc.id for h in index.search("scd2", kind=Kind.STANDARD)] == ["standard:history"]
    assert index.search("nothing matches this") == []
    assert CatalogIndex([]).search("x") == []


def test_benchmark_metrics() -> None:
    queries = QuerySet.model_validate(
        {
            "queries": [
                {"id": "q1", "text": "silver newest", "relevant": ["pattern:silver"]},
                {
                    "id": "q2",
                    "text": "absent words",
                    "relevant": ["pattern:scd2", "pattern:missing"],
                },
            ]
        }
    )
    index = CatalogIndex(
        [doc("pattern:scd2", "SCD2", "history"), doc("pattern:silver", "Silver", "newest row wins")]
    )
    results = run(index, queries)
    assert results[0].recall(1) == 1.0 and results[0].reciprocal_rank == 1.0
    assert results[1].recall(5) == 0.0 and results[1].reciprocal_rank == 0.0
    assert unknown_ids(index, queries) == {"pattern:missing"}
    text = markdown(results, 2, "abc1234")
    assert "| 0.50 | 0.50 | 0.50 | 0.50 |" in text and "- q2: wanted" in text
    assert "None." in markdown([Result(queries.queries[0], ["pattern:silver"])], 2, "x")


def test_the_committed_snapshot_and_queries_agree() -> None:
    docs = read_snapshot(SNAPSHOT)
    assert {d.kind for d in docs} == set(Kind) and len(docs) >= 40
    assert {d.source.commit for d in docs if d.kind is Kind.TABLE} == {"c67e146"}
    queries = read_queries(Path("benchmarks/catalog/queries.yaml"))
    assert len(queries.queries) >= 20 and not unknown_ids(CatalogIndex(docs), queries)


def test_cli(
    lakehouse: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "snap.jsonl"
    assert cli.main(["snapshot", "--lakehouse", str(lakehouse), "--out", str(out)]) == 0
    assert "documents ->" in capsys.readouterr().out
    assert cli.main(["snapshot", "--lakehouse", str(tmp_path)]) == 1

    assert cli.main(["search", "keep customer history", "-k", "3"]) == 0
    assert "pattern:" in capsys.readouterr().out
    assert cli.main(["search", "zzzz qqqq"]) == 0
    assert "no matches" in capsys.readouterr().out
    assert cli.main(["search", "collect", "--kind", "standard"]) == 0
    assert "standard:notebook.no-unbounded-collect-or-topandas" in capsys.readouterr().out

    report = tmp_path / "bench.md"
    assert cli.main(["bench", "--out", str(report)]) == 0
    assert "metadata-driven-lakehouse@c67e146" in report.read_text("utf-8")
    bad = tmp_path / "q.yaml"
    bad.write_text("queries:\n  - {id: x, text: t, relevant: [pattern:nope]}\n", "utf-8")
    assert cli.main(["bench", "--queries", str(bad)]) == 1

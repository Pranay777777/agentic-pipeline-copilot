"""Catalog CLI.

    python -m copilot.catalog snapshot --lakehouse ../metadata-driven-lakehouse
    python -m copilot.catalog search "how do we keep history for customers?" [-k 5] [--kind pattern]
    python -m copilot.catalog bench [--out docs/results/catalog-retrieval.md]

`snapshot` extracts documents from a lakehouse checkout (read statically,
never imported) into catalog/snapshot.jsonl, which is committed: search and
the agents never need the lakehouse itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from copilot.catalog.bench import markdown, read_queries, run, unknown_ids
from copilot.catalog.extract import extract
from copilot.catalog.index import CatalogIndex
from copilot.catalog.model import Kind, read_snapshot, write_snapshot

SNAPSHOT = Path("catalog/snapshot.jsonl")
STANDARDS = Path("catalog/standards.md")
QUERIES = Path("benchmarks/catalog/queries.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot.catalog",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="extract documents from a lakehouse checkout")
    snap.add_argument("--lakehouse", type=Path, required=True)
    snap.add_argument("--out", type=Path, default=SNAPSHOT)
    find = sub.add_parser("search", help="search the catalog")
    find.add_argument("query")
    find.add_argument("-k", type=int, default=5)
    find.add_argument("--kind", choices=[k.value for k in Kind])
    bench = sub.add_parser("bench", help="recall@k and MRR on labelled questions")
    bench.add_argument("--queries", type=Path, default=QUERIES)
    bench.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        if not (args.lakehouse / "src/lakehouse").is_dir():
            print(
                f"error: {args.lakehouse} is not a metadata-driven-lakehouse checkout",
                file=sys.stderr,
            )
            return 1
        docs = extract(args.lakehouse, STANDARDS)
        write_snapshot(args.out, docs)
        counts = {k.value: sum(d.kind is k for d in docs) for k in Kind}
        print(f"{len(docs)} documents -> {args.out} ({counts})")
        return 0

    docs = read_snapshot(SNAPSHOT)
    index = CatalogIndex(docs)
    if args.command == "search":
        hits = index.search(args.query, k=args.k, kind=Kind(args.kind) if args.kind else None)
        for hit in hits:
            print(f"{hit.score:6.2f}  {hit.doc.id:<32} {hit.doc.title}")
        if not hits:
            print("no matches")
        return 0

    queries = read_queries(args.queries)
    missing = unknown_ids(index, queries)
    if missing:
        print(
            f"error: queries cite unknown documents: {', '.join(sorted(missing))}", file=sys.stderr
        )
        return 1
    commit = next((d.source.commit for d in docs if d.source.commit != "HEAD"), "?")
    text = markdown(run(index, queries), len(docs), commit)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

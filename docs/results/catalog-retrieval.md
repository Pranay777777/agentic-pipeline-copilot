# Catalog retrieval - 2026-10-01

24 labelled questions over 49 catalog documents extracted from metadata-driven-lakehouse@c67e146; BM25.

| Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| 0.60 | 0.85 | 0.94 | 0.84 |

## Misses at 5

- q03: wanted pattern:module.transform-silver; got standard:notebook.silver-keeps-one-current-row-per-key, pattern:adr.0005, pattern:module.transform-scd2, pattern:module.transform-gold, pattern:module.ingest-cdc
- q07: wanted pattern:module.quality-quarantine, pattern:adr.0010; got standard:notebook.quality-rules-run-before-writes, pattern:module.quality-engine, pattern:adr.0009, pattern:module.quality-quarantine, pattern:adr.0016

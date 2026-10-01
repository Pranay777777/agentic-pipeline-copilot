"""BM25 search over catalog documents.

BM25 first, on purpose: catalog questions are full of exact identifiers -
`customer_id`, `scd2`, `MERGE`, `dq_rule` - which lexical matching handles
well and small embedding models blur. The tokenizer keeps snake_case
identifiers whole *and* split, so `customer_id` matches both `customer_id`
and "customer id". Dense retrieval can be fused in later if the benchmark
shows misses it would fix (ADR-0003).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from copilot.catalog.model import CatalogDoc, Kind

K1, B = 1.5, 0.75
TITLE_WEIGHT = 3
_WORD = re.compile(r"[a-z0-9]+(?:[_.][a-z0-9]+)*")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "which",
        "with",
        "we",
        "you",
    }
)


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text.casefold()):
        if word in STOPWORDS:
            continue
        out.append(word)
        parts = re.split(r"[_.]", word)
        if len(parts) > 1:
            out.extend(p for p in parts if p and p not in STOPWORDS)
    return out


@dataclass(frozen=True)
class Hit:
    doc: CatalogDoc
    score: float


class CatalogIndex:
    def __init__(self, docs: Sequence[CatalogDoc]) -> None:
        self.docs = list(docs)
        self._tf = [
            Counter(tokens(f"{d.title} " * TITLE_WEIGHT + d.text + " " + " ".join(d.tags)))
            for d in self.docs
        ]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5, kind: Kind | None = None) -> list[Hit]:
        terms = tokens(query)
        scored = []
        for doc, tf, length in zip(self.docs, self._tf, self._len, strict=True):
            if kind is not None and doc.kind is not kind:
                continue
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = f + K1 * (1 - B + B * length / (self._avg or 1))
                    score += self._idf.get(term, 0.0) * f * (K1 + 1) / norm
            if score > 0:
                scored.append(Hit(doc, score))
        scored.sort(key=lambda h: (-h.score, h.doc.id))
        return scored[:k]

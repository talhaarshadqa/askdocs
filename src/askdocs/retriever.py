"""A small, dependency-free TF-IDF retriever."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "it", "this", "that", "with", "as", "at", "by",
}


def _norm(t: str) -> str:
    # light plural/verb normalization so "apis" matches "api", "days" matches "day"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_norm(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    doc_id: str
    chunk_id: int
    text: str
    score: float


class Retriever:
    """Index chunks and return the most relevant ones for a query."""

    def __init__(self) -> None:
        self._chunks: list[tuple[str, int, str, Counter]] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._dirty = False

    def add(self, doc_id: str, chunks: list[str]) -> None:
        for i, text in enumerate(chunks):
            tf = Counter(tokenize(text))
            if not tf:
                continue
            self._chunks.append((doc_id, i, text, tf))
            for term in tf:
                self._df[term] += 1
        self._dirty = True

    def remove(self, doc_id: str) -> int:
        before = len(self._chunks)
        kept = [c for c in self._chunks if c[0] != doc_id]
        removed = before - len(kept)
        if removed:
            self._chunks = kept
            self._df = Counter()
            for _, _, _, tf in self._chunks:
                for term in tf:
                    self._df[term] += 1
            self._dirty = True
        return removed

    def _ensure_idf(self) -> None:
        if not self._dirty:
            return
        n = max(1, len(self._chunks))
        self._idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in self._df.items()}
        self._dirty = False

    def _vec(self, tf: Counter) -> dict[str, float]:
        total = sum(tf.values()) or 1
        return {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, *, k: int = 4) -> list[Hit]:
        self._ensure_idf()
        q = self._vec(Counter(tokenize(query)))
        scored = []
        for doc_id, cid, text, tf in self._chunks:
            s = self._cosine(q, self._vec(tf))
            if s > 0:
                scored.append(Hit(doc_id, cid, text, round(s, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._chunks)

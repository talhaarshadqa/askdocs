"""In-memory document store wired to the retriever."""
from __future__ import annotations

from .chunk import chunk_text
from .retriever import Retriever


class DocStore:
    def __init__(self) -> None:
        self._docs: dict[str, str] = {}
        self.retriever = Retriever()

    def add(self, doc_id: str, text: str) -> int:
        if doc_id in self._docs:
            self.retriever.remove(doc_id)
        self._docs[doc_id] = text
        chunks = chunk_text(text)
        self.retriever.add(doc_id, chunks)
        return len(chunks)

    def delete(self, doc_id: str) -> bool:
        if doc_id not in self._docs:
            return False
        del self._docs[doc_id]
        self.retriever.remove(doc_id)
        return True

    def ids(self) -> list[str]:
        return sorted(self._docs)

    def __len__(self) -> int:
        return len(self._docs)

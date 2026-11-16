"""FastAPI layer over the document store."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .backends import get_backend
from .store import DocStore


class Document(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Question(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


def create_app(store: DocStore | None = None, backend_name: str | None = None) -> FastAPI:
    store = store or DocStore()
    backend = get_backend(backend_name)
    app = FastAPI(title="askdocs", version=__version__,
                  description="Ask questions about your documents (RAG).")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": backend.name, "documents": len(store)}

    @app.put("/documents")
    def upsert(doc: Document) -> dict:
        n = store.add(doc.id, doc.text)
        return {"id": doc.id, "chunks": n}

    @app.get("/documents")
    def list_docs() -> dict:
        return {"ids": store.ids()}

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete(doc_id: str) -> None:
        if not store.delete(doc_id):
            raise HTTPException(404, "document not found")

    @app.post("/ask")
    def ask(q: Question) -> dict:
        hits = store.retriever.search(q.question, k=q.k)
        return {
            "answer": backend.answer(q.question, hits),
            "sources": [{"id": h.doc_id, "chunk": h.chunk_id, "score": h.score} for h in hits],
        }

    return app


app = create_app()

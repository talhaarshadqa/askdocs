"""askdocs - a document Q&A API (RAG) that runs locally or with OpenAI."""
from .chunk import chunk_text
from .retriever import Retriever
from .store import DocStore
from .backends import LocalBackend, get_backend

__all__ = ["chunk_text", "Retriever", "DocStore", "LocalBackend", "get_backend"]
__version__ = "1.0.0"

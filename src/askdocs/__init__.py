"""askdocs - a document Q&A API (RAG) that runs locally or with OpenAI."""
from .backends import LocalBackend, get_backend
from .chunk import chunk_text
from .retriever import Retriever
from .store import DocStore

__all__ = ["DocStore", "LocalBackend", "Retriever", "chunk_text", "get_backend"]
__version__ = "1.0.0"

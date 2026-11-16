"""Answer backends: a zero-dependency local one, and an optional OpenAI one."""
from __future__ import annotations
import os
import textwrap
from typing import Protocol

from .retriever import Hit


class Backend(Protocol):
    def answer(self, question: str, hits: list[Hit]) -> str: ...


class LocalBackend:
    """Extractive answer: stitches the most relevant chunks together.

    No API key, no network - good for demos, tests and offline use.
    """

    name = "local"

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "I couldn't find anything about that in the provided documents."
        top = hits[: 2]
        joined = " [...] ".join(textwrap.shorten(h.text, width=400) for h in top)
        return f"Based on your documents: {joined}"


class OpenAIBackend:
    """Uses OpenAI to write a grounded answer from the retrieved chunks."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def answer(self, question: str, hits: list[Hit]) -> str:  # pragma: no cover
        from openai import OpenAI  # imported lazily so the dep is optional

        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        prompt = (
            "Answer the question using only the context. If the answer is not "
            "in the context, say so. Cite sources like [1].\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


def get_backend(name: str | None = None) -> Backend:
    """Pick a backend. Defaults to OpenAI when a key is set, else local."""
    name = name or os.getenv("ASKDOCS_BACKEND")
    if name == "openai" or (name is None and os.getenv("OPENAI_API_KEY")):
        return OpenAIBackend()
    return LocalBackend()

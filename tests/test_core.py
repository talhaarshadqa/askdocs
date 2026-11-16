import pytest
from askdocs.chunk import chunk_text
from askdocs.retriever import Retriever, tokenize
from askdocs.store import DocStore
from askdocs.backends import LocalBackend, get_backend


def test_chunk_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, size=120, overlap=30)
    assert len(chunks) >= 3
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("The quick Brown fox")
    assert tokenize("Django and FastAPI") == ["django", "fastapi"]


def test_retriever_ranks_relevant_first():
    r = Retriever()
    r.add("d1", ["FastAPI is a Python web framework for building APIs"])
    r.add("d2", ["Pandas is a library for data analysis and dataframes"])
    hits = r.search("python api framework")
    assert hits and hits[0].doc_id == "d1"
    assert hits[0].score > 0


def test_retriever_remove():
    r = Retriever()
    r.add("d1", ["alpha beta gamma"])
    r.add("d2", ["delta epsilon"])
    assert len(r) == 2
    assert r.remove("d1") == 1
    assert all(h.doc_id != "d1" for h in r.search("alpha"))


def test_store_upsert_and_delete():
    s = DocStore()
    s.add("doc", "web scraping with python and beautifulsoup " * 20)
    assert len(s) == 1
    s.add("doc", "updated content about automation " * 20)  # re-index
    assert len(s) == 1
    assert s.delete("doc") is True
    assert s.delete("missing") is False


def test_local_backend():
    b = LocalBackend()
    assert "couldn't find" in b.answer("q", [])
    s = DocStore()
    s.add("d", "The deployment runs on Docker with a Postgres database.")
    hits = s.retriever.search("what docker and database setup is used")
    out = b.answer("what docker and database setup is used", hits)
    assert "Docker" in out


def test_get_backend_defaults_local(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ASKDOCS_BACKEND", raising=False)
    assert get_backend().name == "local"
    assert get_backend("openai").name == "openai"

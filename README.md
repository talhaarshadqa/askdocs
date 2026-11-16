# 🤖 askdocs

[![CI](https://github.com/talhaarshadqa/askdocs/actions/workflows/ci.yml/badge.svg)](https://github.com/talhaarshadqa/askdocs/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

**Ask questions about your own documents — a small RAG (retrieval-augmented generation) API.**
Point it at your docs, ask a question, and get an answer with the sources it came from.

Works **with zero API key** out of the box (a built-in TF-IDF retriever + extractive answers),
and upgrades to **OpenAI** for natural-language answers when you add a key.

## ✨ Features

- **Retrieval-augmented**: answers are grounded in your documents, with source references
- **No-key local mode**: dependency-free TF-IDF retriever, great for demos and tests
- **OpenAI mode**: set `OPENAI_API_KEY` for fluent, cited answers (`gpt-4o-mini` by default)
- **Simple REST API**: upload docs, ask questions, delete docs
- **Automatic chunking** with overlap, so answers don't get cut at boundaries
- Tested, typed, and CI-checked on Python 3.10–3.13

## 🚀 Quick start

```bash
git clone https://github.com/talhaarshadqa/askdocs.git
cd askdocs
pip install -e .
uvicorn askdocs.api:app --reload
```

```bash
# add a document
curl -X PUT localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"id":"faq","text":"We ship worldwide in 3 to 5 business days via DHL."}'

# ask about it
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"how long does delivery take?"}'
```

```json
{
  "answer": "Based on your documents: We ship worldwide in 3 to 5 business days via DHL.",
  "sources": [{"id": "faq", "chunk": 0, "score": 0.52}]
}
```

### Use OpenAI for natural answers

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=sk-...
uvicorn askdocs.api:app          # answers now written by the model, still cited
```

## 📡 API

| Method | Endpoint | Description |
|---|---|---|
| `PUT` | `/documents` | Add or replace a document (`{id, text}`) |
| `GET` | `/documents` | List document IDs |
| `DELETE` | `/documents/{id}` | Remove a document |
| `POST` | `/ask` | Ask a question (`{question, k}`) → answer + sources |
| `GET` | `/health` | Health + active backend |

## 🧠 How it works

```
documents ──chunk──► TF-IDF index ──search(query)──► top-k chunks ──► backend ──► answer + sources
```

The retriever and chunker are pure Python (no heavy ML deps), so the core runs anywhere
and is fully unit-tested. The answer backend is pluggable — swap `local` for `openai`,
or add your own.

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest --cov=askdocs
```

## 📄 License

MIT

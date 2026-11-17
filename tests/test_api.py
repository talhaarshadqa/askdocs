import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from askdocs.api import create_app
from askdocs.store import DocStore


@pytest.fixture
def client():
    return TestClient(create_app(DocStore(), backend_name="local"))


def test_flow(client):
    assert client.get("/health").json()["backend"] == "local"
    r = client.put("/documents", json={"id": "faq", "text": "We ship worldwide in 3 to 5 business days via DHL."})
    assert r.status_code == 200 and r.json()["chunks"] >= 1
    a = client.post("/ask", json={"question": "how many business days worldwide delivery"}).json()
    assert "DHL" in a["answer"] or "business days" in a["answer"]
    assert a["sources"][0]["id"] == "faq"
    assert client.get("/documents").json()["ids"] == ["faq"]
    assert client.delete("/documents/faq").status_code == 204
    assert client.delete("/documents/faq").status_code == 404


def test_validation(client):
    assert client.put("/documents", json={"id": "", "text": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "k": 0}).status_code == 422

import os
import tempfile

os.environ["VECTOR_DB_PATH"] = tempfile.mkdtemp()
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient

from app.main import app, pipeline
from app.schemas import AskResponse

client = TestClient(app)


def test_health():
    response = client.get("/health").json()
    assert response["status"] == "ok"
    assert response["cache"] == "local-fallback"


def test_upload_rejects_bad_type():
    assert client.post("/upload", files={"files": ("x.exe", b"hi")}).status_code == 400


def test_upload_and_ask_with_fake_llm(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "embed", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(pipeline.llm, "answer", lambda q, ctx: AskResponse(answer=ctx[0]["text"], sources=[ctx[0]["source"]], confidence=0.9))
    r = client.post("/upload", files={"files": ("notes.txt", b"vLLM serves models fast.")})
    assert r.status_code == 200 and r.json()["uploaded"][0]["chunks"] == 1
    a = client.post("/ask", json={"question": "what is vLLM?"}).json()
    assert a["sources"] == ["notes.txt"] and a["confidence"] == 0.9
    meta = pipeline.store.document_metadata("notes.txt")
    assert meta["chunk_count"] == 1 and meta["page_count"] is None

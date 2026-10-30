from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.api.conftest import FakeMultimodalRAG


def test_health_reports_unloaded_services(client):
    body = client.get("/api/health").json()
    assert body["status"] == "operational"
    assert set(body["services"]) == {"guard_agent", "rag_engine", "advisor_agent", "dna_agent"}
    assert all(v.startswith("unavailable") for v in body["services"].values())


def test_health_shows_advisor_load_error(client, state):
    state.advisor_load_error = "dataset missing"
    assert client.get("/api/health").json()["services"]["advisor_agent"] == "unavailable: dataset missing"


def test_liveness_probe(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_responses_carry_request_id(client):
    r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/api/rag/query", {"query": "x"}),
        ("post", "/api/rag/chat", {"message": "x"}),
        ("post", "/api/advisor/chat", {"user_id": "U", "message": "x"}),
        ("get", "/api/advisor/users", None),
        ("post", "/api/security/chat", {"message": "x"}),
        ("get", "/api/dna/profile/U", None),
        ("post", "/api/dna/compare", {"user_id": "U"}),
    ],
)
def test_routes_return_503_when_service_missing(client, method, path, body):
    r = getattr(client, method)(path, json=body) if body else getattr(client, method)(path)
    assert r.status_code == 503


def test_login_success_and_failure(client, state):
    ok = client.post("/api/auth/login", json={"username": "analyst", "password": "analyst123"}).json()
    assert ok["authenticated"] is True and ok["session_id"] and ok["diverted"] is False
    bad = client.post("/api/auth/login", json={"username": "analyst", "password": "wrong"}).json()
    assert bad["authenticated"] is False and bad["session_id"] is None
    assert state.login_failures["analyst"] == 1


def test_validation_error_is_422(client):
    assert client.post("/api/rag/query", json={"query": "x", "n_results": 99}).status_code == 422


def test_unhandled_errors_do_not_leak_internals(client, state):
    def explode(*a, **k):
        raise RuntimeError("secret internal path /Users/x")

    state.guard_agent = SimpleNamespace(analyze=explode)
    r = client.post("/api/security/chat", json={"message": "hi"})
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["request_id"]


def test_security_chat_passes_approval_through(client, state):
    seen = {}

    def analyze(message, session_id=None, approved=False):
        seen["approved"] = approved
        return {"reply": "ok", "actions": [{"step": 1, "tool": "t", "args": {}, "result": "r"}, "junk"]}

    state.guard_agent = SimpleNamespace(analyze=analyze)
    body = client.post("/api/security/chat", json={"message": "hi", "session_id": "s1"}).json()
    assert seen["approved"] is False
    assert body["reply"] == "ok" and body["session_id"] == "s1" and len(body["actions"]) == 1


def test_advisor_chat_maps_result(client, state):
    state.advisor_agent = SimpleNamespace(
        chat=lambda m, u, session_id=None, approved=False: {"reply": f"{u}:{m}", "tools_used": ["x"]}
    )
    body = client.post("/api/advisor/chat", json={"user_id": "U1", "message": "hi"}).json()
    assert body["reply"] == "U1:hi" and body["tool_results"] == [] and body["tools_used"] == ["x"]


def test_dna_challenge_requires_approval(client):
    body = client.post("/api/dna/challenge", json={"user_id": "U1", "approved": False}).json()
    assert body["status"] == "pending_approval" and body["challenge"] is None


def test_dna_compare_low_trust_needs_approval(client, state):
    result = {
        "user_id": "U1",
        "baseline_radar": [1.0],
        "session_radar": [0.1],
        "radar_labels": ["a"],
        "axis_deviations": {"a": 0.9},
        "composite_deviation": 0.9,
        "session_trust_score": 0.2,
        "verdict": "anomalous",
    }
    state.dna_agent = SimpleNamespace(compare_session=lambda u, session_overrides=None: dict(result))
    body = client.post("/api/dna/compare", json={"user_id": "U1"}).json()
    assert body["requires_approval"] is True and body["challenge"] is None


# --- uploads ----------------------------------------------------------------


def _upload(client, files, session_id=None):
    params = {"session_id": session_id} if session_id else {}
    return client.post("/api/rag/upload", params=params, files=[("files", f) for f in files])


def test_upload_stores_and_indexes_per_session(client, state, uploads_dir):
    state.multimodal_rag = FakeMultimodalRAG()
    r = _upload(client, [("notes.txt", b"hello", "text/plain")], session_id="sessA")
    assert r.status_code == 200
    entry = r.json()["uploads"][0]
    assert entry["status"] == "indexed successfully" and entry["index"]["status"] == "indexed"
    assert (uploads_dir / "sessA" / "notes.txt").read_bytes() == b"hello"
    assert state.multimodal_rag.indexed == [("sessA", "notes.txt", 5)]


def test_upload_rejects_unsupported_type(client, uploads_dir):
    entry = _upload(client, [("run.exe", b"MZ", "application/octet-stream")], session_id="s").json()[
        "uploads"
    ][0]
    assert entry["status"] == "skipped"
    assert not (uploads_dir / "s" / "run.exe").exists()


def test_upload_filename_cannot_escape_session_dir(client, uploads_dir):
    entry = _upload(client, [("../../evil.txt", b"x", "text/plain")], session_id="s").json()["uploads"][0]
    assert entry["filename"] == "evil.txt"
    assert (uploads_dir / "s" / "evil.txt").exists()
    assert not (uploads_dir.parent / "evil.txt").exists()


def test_upload_rejects_path_like_session_id(client, uploads_dir):
    assert _upload(client, [("a.txt", b"x", "text/plain")], session_id="../other").status_code == 400


def test_upload_enforces_size_limit(client, uploads_dir, monkeypatch):
    import api.uploads as up
    from core.settings import Settings

    monkeypatch.setattr(up, "get_settings", lambda: Settings(max_upload_bytes=4))
    entry = _upload(client, [("big.txt", b"12345", "text/plain")], session_id="s").json()["uploads"][0]
    assert entry["status"] == "skipped" and entry["error"] == "File too large."


def test_doc_chat_is_isolated_between_sessions(client, state, uploads_dir):
    state.multimodal_rag = FakeMultimodalRAG()
    _upload(client, [("a.txt", b"Alpha secret balance 4242", "text/plain")], session_id="sessA")
    _upload(client, [("b.txt", b"Bravo notes", "text/plain")], session_id="sessB")

    body = client.post("/api/rag/chat", json={"message": "balance?", "session_id": "sessB"}).json()
    filenames = {s["filename"] for s in body["sources"]}
    assert filenames == {"b.txt"}
    assert "4242" not in body["reply"]

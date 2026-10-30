"""API test fixtures: the real app with lifespan model loading bypassed and fake services."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import api.uploads as uploads_module
from api.main import create_app
from api.state import get_state


class FakeMultimodalRAG:
    """In-memory, session-scoped stand-in for ``models.multimodal_rag.MultimodalRAG``."""

    def __init__(self) -> None:
        self.docs: Dict[str, List[Dict[str, Any]]] = {}
        self.indexed: List[tuple] = []

    def index_file_bytes(self, filename: str, data: bytes, session_id: str = "global") -> dict:
        self.indexed.append((session_id, filename, len(data)))
        self.docs.setdefault(session_id, []).append(
            {
                "text": data.decode("utf-8", "ignore"),
                "confidence": 0.9,
                "metadata": {"filename": filename, "session_id": session_id},
                "filename": filename,
            }
        )
        return {"filename": filename, "status": "indexed", "chunks": 1}

    def query(self, message, n_results=6, include_types=None, session_id=None):
        return list(self.docs.get(session_id, []))[:n_results]

    def force_image_hits(self, session_id):
        return []

    def force_document_hits(self, session_id):
        return []

    def needs_vision(self, message, results, session_id):
        return False

    def resolve_session_image_paths(self, session_id):
        return []

    def live_analyze_session_images(self, session_id, message):
        return {}

    def get_file_inventory(self, session_id):
        return [{"filename": d["filename"]} for d in self.docs.get(session_id, [])]


@pytest.fixture
def state():
    s = get_state()
    s.reset()
    yield s
    s.reset()


@pytest.fixture
def client(state):
    # No ``with`` block → the lifespan (real model loading) never runs.
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture
def uploads_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads_module, "UPLOADS_DIR", tmp_path)
    return tmp_path

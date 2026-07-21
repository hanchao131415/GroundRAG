from fastapi.testclient import TestClient

from app.auth import create_access_token


def _auth_headers():
    from config import DEFAULT_CONFIG

    token = create_access_token("zhangsan", secret=DEFAULT_CONFIG.jwt_secret)
    return {"Authorization": f"Bearer {token}"}


def test_health_is_lightweight_and_does_not_initialize_rag(monkeypatch):
    from app import api

    monkeypatch.setattr(api, "_rag", None)
    monkeypatch.setattr(api, "_rag_status", "initializing", raising=False)
    monkeypatch.setattr(
        api,
        "_initialize_rag_sync",
        lambda: (_ for _ in ()).throw(AssertionError("health initialized RAG")),
        raising=False,
    )

    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_while_initializing(monkeypatch):
    from app import api

    monkeypatch.setattr(api, "_rag", None)
    monkeypatch.setattr(api, "_rag_status", "initializing", raising=False)
    monkeypatch.setattr(api, "_rag_error", None, raising=False)

    response = TestClient(api.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "RAG_NOT_READY"
    assert response.json()["detail"]["status"] == "initializing"


def test_search_returns_structured_503_when_rag_failed(monkeypatch):
    from app import api

    monkeypatch.setattr(api, "_rag", None)
    monkeypatch.setattr(api, "_rag_status", "error", raising=False)
    monkeypatch.setattr(api, "_rag_error", "model unavailable", raising=False)

    response = TestClient(api.app).post(
        "/api/v1/search",
        json={"question": "q"},
        headers=_auth_headers(),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "RAG_NOT_READY",
        "status": "error",
        "message": "model unavailable",
    }


def test_sse_bridge_has_bounded_queue_and_cancellation():
    from app.api import SSEBridge

    bridge = SSEBridge(lambda: iter(()), maxsize=2)

    assert bridge.queue.maxsize == 2
    assert bridge.cancelled.is_set() is False
    bridge.cancel()
    assert bridge.cancelled.is_set() is True

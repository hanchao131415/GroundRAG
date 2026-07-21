from types import SimpleNamespace

from fastapi.testclient import TestClient


class _Users:
    users = {
        "zhangsan": {"name": "张三", "departments": ["HR"], "role": "员工"},
        "admin": {"name": "管理员", "departments": ["*"], "role": "管理员"},
    }

    def authenticate(self, uid):
        return self.users.get(uid)

    def get_departments(self, uid):
        return self.users[uid]["departments"]


class _Rag:
    user_service = _Users()
    cache = None


def _client(monkeypatch, tmp_path):
    from app import api

    monkeypatch.setattr(api, "_rag", _Rag())
    monkeypatch.setattr(api, "_rag_status", "ready")
    monkeypatch.setattr(api, "_reindex_task", None)
    monkeypatch.setattr(api, "DEFAULT_CONFIG", SimpleNamespace(
        data_path=str(tmp_path), jwt_secret="test-secret", jwt_expire_hours=1,
    ))
    monkeypatch.setattr(api, "_schedule_reindex", lambda: None)
    return TestClient(api.app)


def _token(client, user_id):
    from app.auth import create_access_token
    from config import DEFAULT_CONFIG

    return create_access_token(user_id, secret=DEFAULT_CONFIG.jwt_secret)


def test_upload_list_and_delete_document(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _token(client, "zhangsan")
    headers = {"Authorization": f"Bearer {token}"}

    uploaded = client.post(
        "/api/v1/documents?department=HR",
        headers=headers,
        files={"file": ("policy.md", b"# annual leave", "text/markdown")},
    )
    assert uploaded.status_code == 202
    assert uploaded.json()["index_status"] == "reindexing"
    document = uploaded.json()["document"]
    assert document["department"] == "HR"
    assert (tmp_path / "HR" / "policy.md").read_bytes() == b"# annual leave"

    listed = client.get("/api/v1/documents", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["id"] == document["id"]

    removed = client.delete(f"/api/v1/documents/{document['id']}", headers=headers)
    assert removed.status_code == 202
    assert removed.json()["index_status"] == "reindexing"
    assert not (tmp_path / "HR" / "policy.md").exists()


def test_upload_rejects_unauthorized_department(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _token(client, "zhangsan")

    response = client.post(
        "/api/v1/documents?department=FIN",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("expense.md", b"secret", "text/markdown")},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "DOCUMENT_FORBIDDEN"


def test_reindex_requires_admin(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    token = _token(client, "zhangsan")

    response = client.post("/api/v1/documents/reindex", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "ADMIN_REQUIRED"


def test_admin_reindex_can_schedule_on_request_event_loop(monkeypatch, tmp_path):
    from app import api

    original_schedule = api._schedule_reindex
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "_schedule_reindex", original_schedule)
    monkeypatch.setattr(api, "_initialize_rag_sync", _Rag)
    monkeypatch.setattr(api, "_reindex_task", None)
    token = _token(client, "admin")

    response = client.post(
        "/api/v1/documents/reindex",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    assert response.json()["index_status"] == "reindexing"


def test_delete_can_schedule_on_request_event_loop(monkeypatch, tmp_path):
    from app import api
    from app.document_service import save_upload

    original_schedule = api._schedule_reindex
    client = _client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "_schedule_reindex", original_schedule)
    monkeypatch.setattr(api, "_initialize_rag_sync", _Rag)
    monkeypatch.setattr(api, "_reindex_task", None)
    document = save_upload(str(tmp_path), "policy.md", b"policy", "HR")
    token = _token(client, "zhangsan")

    response = client.delete(
        f"/api/v1/documents/{document.document_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    assert response.json()["index_status"] == "reindexing"

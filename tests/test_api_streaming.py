import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient


def _client(monkeypatch=None):
    from app import api
    class FakeUserService:
        users = {
            "zhangsan": {"name": "张三", "departments": ["HR"], "role": "员工"},
            "admin": {"name": "管理员", "departments": ["*"], "role": "管理员"},
        }
        def authenticate(self, uid): return self.users.get(uid)
        def get_departments(self, uid): return self.users[uid]["departments"]
    class FakeRag:
        user_service = FakeUserService()
    api._rag = FakeRag()
    return TestClient(api.app)


def test_demo_users_returns_list():
    client = _client()
    resp = client.get("/auth/demo-users")
    assert resp.status_code == 200
    data = resp.json()
    uids = {u["user_id"] for u in data["users"]}
    assert "zhangsan" in uids and "admin" in uids
    zs = next(u for u in data["users"] if u["user_id"] == "zhangsan")
    assert zs["name"] == "张三" and zs["departments"] == ["HR"] and zs["role"] == "员工"

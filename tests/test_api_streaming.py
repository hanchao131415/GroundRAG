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


def test_ask_stream_emits_typed_events_in_order():
    """ask_stream 必须按序发 sources→token(s)→trace→done，且 trace 含各步。"""
    from main import EnterpriseRAGSystem
    from langchain_core.documents import Document

    sys_obj = EnterpriseRAGSystem.__new__(EnterpriseRAGSystem)
    sys_obj.config = type("C", (), {"llm_provider": "deepseek", "top_k": 3})()
    sys_obj.cache = None
    sys_obj.current_user = None

    fake_chunk = Document(page_content="工作满3年年假10天。", metadata={
        "source": "HR/年假管理制度.md", "page": 1, "department": "HR", "vector_sim": 0.9})

    class FakeRet:
        def permission_aware_search(self, q, deps, top_k): return [fake_chunk]
        def hybrid_search(self, q, top_k): return [fake_chunk]
    class FakeGen:
        ANSWER = "工作满3年年假10天。"
        def query_router(self, q): return "retrieval"
        def query_rewrite(self, q): return q
        def generate_answer_stream_with_usage(self, q, chunks):
            yield self.ANSWER[:6], None
            yield self.ANSWER[6:], {"prompt": 10, "completion": 5, "total": 15}
    sys_obj.retrieval_module = FakeRet()
    sys_obj.generation_module = FakeGen()

    events = list(sys_obj.ask_stream("工作满3年年假几天", user_departments=["HR"], user_id="zhangsan"))
    types = [e["type"] for e in events]

    assert types[0] == "sources"                      # sources first
    assert types[-1] == "done"                         # done last
    assert "trace" in types                            # trace present
    assert types.index("sources") < types.index("trace") < types.index("done")
    # sources content
    src = next(e for e in events if e["type"] == "sources")["items"][0]
    assert src["source"] == "HR/年假管理制度.md" and src["department"] == "HR"
    # tokens concatenate to the answer
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert answer == "工作满3年年假10天。"
    # trace has steps + tokens + cost
    tr = next(e for e in events if e["type"] == "trace")["trace"]
    assert tr["trace_id"] and tr["total_ms"] >= 0
    assert tr["tokens"]["total"] == 15
    assert any(s["name"] == "检索" for s in tr["steps"])
    assert tr["cost_usd"] >= 0


def test_ask_stream_no_result_refuses():
    """检索为空时返回拒答 token + trace + done。"""
    from main import EnterpriseRAGSystem
    sys_obj = EnterpriseRAGSystem.__new__(EnterpriseRAGSystem)
    sys_obj.config = type("C", (), {"llm_provider": "deepseek", "top_k": 3})()
    sys_obj.cache = None
    sys_obj.current_user = None
    class FakeRet:
        def hybrid_search(self, q, top_k): return []
    class FakeGen:
        def query_router(self, q): return "retrieval"
        def query_rewrite(self, q): return q
    sys_obj.retrieval_module = FakeRet()
    sys_obj.generation_module = FakeGen()

    events = list(sys_obj.ask_stream("无关问题", user_departments=None))
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "未找到" in answer


def test_chat_non_stream_returns_sources_and_trace():
    """非流式 /chat 返回真实 sources + trace（不再空）。"""
    from app import api
    from main import EnterpriseRAGSystem

    class FakeUserService:
        users = {"zhangsan": {"name": "张三", "departments": ["HR"], "role": "员工"}}
        def authenticate(self, uid): return self.users.get(uid)
        def get_departments(self, uid): return self.users[uid]["departments"]

    class FakeRag(EnterpriseRAGSystem):
        def __init__(self):
            self.user_service = FakeUserService()
            self.config = type("C", (), {"top_k": 3, "jwt_secret": "x"})()
        def ask_stream(self, question, user_departments=None, user_id=None):
            yield {"type": "sources", "items": [{"source": "HR/x.md", "page": 1, "department": "HR", "score": 0.9, "preview": "p"}]}
            yield {"type": "token", "text": "答案"}
            yield {"type": "trace", "trace": {"trace_id": "abc", "steps": [], "total_ms": 5,
                                              "tokens": {"prompt": 1, "completion": 1, "total": 2}, "cost_usd": 0.0}}
            yield {"type": "done"}

    api._rag = FakeRag()
    client = TestClient(api.app)
    token = client.post("/auth/login", json={"user_id": "zhangsan"}).json()["access_token"]
    resp = client.post("/api/v1/chat", json={"question": "q", "stream": False},
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "答案"
    assert body["sources"][0]["source"] == "HR/x.md"
    assert body["trace"]["trace_id"] == "abc"
    assert body["trace_id"] == "abc"


def test_chat_stream_emits_sse_frames():
    from app import api
    from main import EnterpriseRAGSystem

    class FakeUserService:
        users = {"zhangsan": {"name": "张三", "departments": ["HR"], "role": "员工"}}
        def authenticate(self, uid): return self.users.get(uid)
        def get_departments(self, uid): return self.users[uid]["departments"]
    class FakeRag(EnterpriseRAGSystem):
        def __init__(self):
            self.user_service = FakeUserService()
            self.config = type("C", (), {"top_k": 3, "jwt_secret": "x"})()
        def ask_stream(self, question, user_departments=None, user_id=None):
            yield {"type": "sources", "items": []}
            yield {"type": "token", "text": "hi"}
            yield {"type": "done"}

    api._rag = FakeRag()
    client = TestClient(api.app)
    token = client.post("/auth/login", json={"user_id": "zhangsan"}).json()["access_token"]
    resp = client.post("/api/v1/chat", json={"question": "q", "stream": True},
                       headers={"Authorization": f"Bearer {token}"})
    body = resp.text
    assert "data:" in body
    assert '"type": "sources"' in body or '"type":"sources"' in body
    assert '"type": "token"' in body or '"type":"token"' in body
    assert "data: [DONE]" in body

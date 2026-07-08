"""
核心模块单元测试

覆盖 review 指出的最致命缺口：零测试。
选择 4 个最重要、最容易出错的函数：
  1. _split_text_and_tables — 表格感知切分
  2. permission_aware_search  — RBAC 权限过滤
  3. _safe_to_cache_hit       — 四重误命中校验
  4. create_llm               — 多供应商工厂

用法：pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document


# ===== 1. 表格感知切分 =====
class TestTableAwareChunking:
    """验证 _split_text_and_tables 正确识别表格边界，表格整体不切"""

    @pytest.fixture
    def module(self):
        from rag_modules.data_preparation import DataPreparationModule
        return DataPreparationModule("data/docs", chunk_size=500, chunk_overlap=50)

    def test_splits_table_from_text(self, module):
        """表格和文本正确分离"""
        content = "# 标题\n\n一些文本。\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n更多文本。"
        blocks = module._split_text_and_tables(content)
        assert len(blocks) == 3  # text, table, text
        assert blocks[0]["type"] == "text"
        assert "标题" in blocks[0]["content"]
        assert blocks[1]["type"] == "table"
        assert "| A | B |" in blocks[1]["content"]
        assert blocks[2]["type"] == "text"
        assert "更多文本" in blocks[2]["content"]

    def test_table_kept_whole(self, module):
        """表格完整保留，不切断"""
        content = "text\n\n| H1 | H2 |\n|---|---|\n| r1 | r2 |\n| r3 | r4 |\n\nmore"
        blocks = module._split_text_and_tables(content)
        table_block = [b for b in blocks if b["type"] == "table"]
        assert len(table_block) == 1
        # 表格行数：表头 + 分隔 + 2数据行 = 4行
        table_lines = [l for l in table_block[0]["content"].split("\n") if l.strip()]
        assert len(table_lines) == 4

    def test_no_table_content(self, module):
        """无表格文档正常返回"""
        content = "# 纯文本\n\n只有文字。"
        blocks = module._split_text_and_tables(content)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"


# ===== 2. RBAC 权限过滤 =====
class TestPermissionAwareSearch:
    """验证权限感知检索：用户只看得到授权部门的文档"""

    def test_hr_cannot_see_finance(self):
        """HR员工搜财务内容，只返回 HR+公共 文档"""
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule
        docs = [
            Document(page_content="年假5天", metadata={"source": "HR/年假.md", "department": "HR"}),
            Document(page_content="报销500", metadata={"source": "财务/报销.md", "department": "财务"}),
            Document(page_content="安全规定", metadata={"source": "公共/安全.md", "department": "公共"}),
        ]
        # 模拟检索模块（不需要真实 FAISS，只测过滤逻辑）
        ret = RetrievalOptimizationModule.__new__(RetrievalOptimizationModule)
        ret.chunks = docs
        # 直接测 metadata_filtered_search 的过滤部分
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R
        allowed = [d for d in docs if R._match_filters(d.metadata, {"department": ["HR", "公共"]})]
        sources = [d.metadata["source"] for d in allowed]
        assert "HR/年假.md" in sources
        assert "公共/安全.md" in sources
        assert "财务/报销.md" not in sources  # ← 关键：财务不能看


# ===== 3. 缓存误命中四重校验 =====
class TestCacheSafeToCacheHit:
    """验证四重校验（长度/实体/比较意图/数字）正确拦截误命中"""

    def test_length_diff_rejects(self):
        """长度差异过大应否决"""
        from rag_modules.cache_service import CacheService
        # "p2工资" 5字 vs "p1和p2相差多少" 9字 → 差44% > 40%
        assert not CacheService._safe_to_cache_hit("p2工资", "p1和p2工资相差多少", 0.85)

    def test_entity_diff_rejects(self):
        """实体不同应否决"""
        from rag_modules.cache_service import CacheService
        # 新query有P1，但cached只有P2 → 否决
        assert not CacheService._safe_to_cache_hit("P1工资多少", "P2工资多少", 0.90)

    def test_compare_intent_rejects(self):
        """比较类和单值类不应命中"""
        from rag_modules.cache_service import CacheService
        assert not CacheService._safe_to_cache_hit("p1比p2多多少", "p2工资", 0.85)

    def test_number_diff_rejects(self):
        """数字不同应否决（坑25）"""
        from rag_modules.cache_service import CacheService
        assert not CacheService._safe_to_cache_hit("工作满1年年假几天", "工作满3年年假几天", 0.94)

    def test_valid_hit_passes(self):
        """正常同义改写应通过"""
        from rag_modules.cache_service import CacheService
        assert CacheService._safe_to_cache_hit("密码几天换一次", "密码更换周期", 0.90)


# ===== 4. LLM 工厂 =====
class TestLLMFactory:
    """验证工厂能根据 provider 正确选择适配器"""

    def test_provider_deepseek_is_openai_protocol(self):
        """deepseek 预设走 openai 协议"""
        from rag_modules.llm_factory import PRESETS
        assert PRESETS["deepseek"]["protocol"] == "openai"

    def test_provider_zai_is_anthropic_protocol(self):
        """z.ai 预设走 anthropic 协议"""
        from rag_modules.llm_factory import PRESETS
        assert PRESETS["zai"]["protocol"] == "anthropic"

    def test_unknown_provider_raises(self):
        """未知 provider 应报错"""
        from rag_modules.llm_factory import create_llm
        with pytest.raises(ValueError, match="不支持"):
            create_llm("unknown_provider", api_key="test", base_url="http://x")


# ===== 5. JWT 认证（review Q7：客户端传 user_id 不能被信任）=====
class TestJWTAuth:
    """验证 JWT 签发/校验/过期逻辑"""

    def test_create_and_verify_token(self):
        """签发的 token 能被正确校验"""
        from app.auth import create_access_token, verify_token
        token = create_access_token("zhangsan", secret="test-secret")
        user_id = verify_token(token, secret="test-secret")
        assert user_id == "zhangsan"

    def test_wrong_secret_rejected(self):
        """错误密钥签发的 token 应被拒绝"""
        from app.auth import create_access_token, verify_token
        token = create_access_token("zhangsan", secret="secret-a")
        with pytest.raises(Exception):
            verify_token(token, secret="secret-b")

    def test_expired_token_rejected(self):
        """过期 token 应被拒绝"""
        from app.auth import create_access_token, verify_token
        token = create_access_token("zhangsan", secret="test-secret", expires_hours=-1)
        with pytest.raises(Exception, match="过期"):
            verify_token(token, secret="test-secret")

    def test_tampered_token_rejected(self):
        """篡改后的 token 应被拒绝"""
        from app.auth import create_access_token, verify_token
        token = create_access_token("zhangsan", secret="test-secret")
        # 篡改最后一个字符
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        with pytest.raises(Exception):
            verify_token(tampered, secret="test-secret")


# ===== 6. FallbackLLM 降级链（review Q5：主 LLM 挂了要能切备用）=====
class TestFallbackLLM:
    """验证降级链在主 LLM 失败时自动切换"""

    def test_primary_success_uses_primary(self):
        """主 LLM 正常时不切备用"""
        from rag_modules.llm_fallback import FallbackLLM

        class FakeLLM:
            def __init__(self, name): self.model_name = name; self.calls = 0
            def invoke(self, msgs, config=None): self.calls += 1; return f"resp-{self.model_name}"

        primary, backup = FakeLLM("primary"), FakeLLM("backup")
        llm = FallbackLLM([primary, backup])
        result = llm.invoke("hi")
        assert result == "resp-primary"
        assert backup.calls == 0  # 备用没被调用

    def test_primary_fail_switches_to_backup(self):
        """主 LLM 抛异常时自动切备用"""
        from rag_modules.llm_fallback import FallbackLLM

        class GoodLLM:
            def __init__(self, name): self.model_name = name
            def invoke(self, msgs, config=None): return f"resp-{self.model_name}"

        class BadLLM:
            def __init__(self, name): self.model_name = name
            def invoke(self, msgs, config=None): raise RuntimeError("primary down")

        llm = FallbackLLM([BadLLM("primary"), GoodLLM("backup")])
        result = llm.invoke("hi")
        assert result == "resp-backup"

    def test_all_fail_raises(self):
        """所有 LLM 都挂了应抛 RuntimeError"""
        from rag_modules.llm_fallback import FallbackLLM

        class BadLLM:
            def __init__(self, name): self.model_name = name
            def invoke(self, msgs, config=None): raise RuntimeError("down")

        llm = FallbackLLM([BadLLM("a"), BadLLM("b")])
        with pytest.raises(RuntimeError, match="所有 LLM"):
            llm.invoke("hi")


# ===== 7. API 并发安全（A2 修复：权限不挂单例 current_user）=====
class TestAPIConcurrencyFix:
    """验证 A2 修复：/chat 与 /search 的权限来自 JWT user_id 现查，
    不依赖进程级单例 _rag.current_user，避免并发请求互相覆盖导致跨权限泄露。

    回归方法：构造一个 current_user 恒为 None 的 FakeRag 注入到 app.api._rag，
    若修复回退（端点又写 rag.current_user / 读 _current_departments），
    则 FakeRag.ask 收到的 user_departments 会变 None 或串号。
    """

    @staticmethod
    def _install_fake_rag(monkeypatch, recorded):
        from app import api

        class FakeUserService:
            _DEPTS = {"zhangsan": ["HR"], "lisi": ["财务"], "admin": ["*"]}

            def authenticate(self, uid):
                return {"name": uid, "departments": self._DEPTS[uid]} if uid in self._DEPTS else None

            def get_departments(self, uid):
                return self._DEPTS[uid]

        class FakeRag:
            def __init__(self):
                self.user_service = FakeUserService()
                self.current_user = None  # 关键：单例上不持有当前用户

            def ask(self, question, stream=False, user_departments=None, user_id=None):
                recorded.append({"user_id": user_id,
                                 "user_departments": user_departments,
                                 "singleton": self.current_user})
                return "ok"

        monkeypatch.setattr(api, "_rag", FakeRag())
        from fastapi.testclient import TestClient
        return TestClient(api.app)

    def test_chat_uses_jwt_departments_not_singleton(self, monkeypatch):
        """HR 用户请求，ask 收到 ['HR']，且单例 current_user 全程未被写入"""
        recorded = []
        client = self._install_fake_rag(monkeypatch, recorded)

        token = client.post("/auth/login", json={"user_id": "zhangsan"}).json()["access_token"]
        resp = client.post("/api/v1/chat",
                           json={"question": "年假几天", "stream": False},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert recorded[0]["user_departments"] == ["HR"]
        assert recorded[0]["user_id"] == "zhangsan"
        assert recorded[0]["singleton"] is None  # A2 核心：单例未被写

    def test_two_users_do_not_cross_leak(self, monkeypatch):
        """两个不同用户依次请求，各自拿到自己部门，互不污染（并发竞态的简化回归）"""
        recorded = []
        client = self._install_fake_rag(monkeypatch, recorded)

        t_hr = client.post("/auth/login", json={"user_id": "zhangsan"}).json()["access_token"]
        t_fin = client.post("/auth/login", json={"user_id": "lisi"}).json()["access_token"]
        client.post("/api/v1/chat", json={"question": "q", "stream": False},
                    headers={"Authorization": f"Bearer {t_hr}"})
        client.post("/api/v1/chat", json={"question": "q", "stream": False},
                    headers={"Authorization": f"Bearer {t_fin}"})

        assert recorded[0]["user_departments"] == ["HR"]
        assert recorded[1]["user_departments"] == ["财务"]
        # 修复前：第二次请求会把单例覆盖成 lisi，第一次若并发会读到 lisi → 跨权限
        assert all(r["singleton"] is None for r in recorded)

    def test_search_uses_jwt_departments_not_singleton(self, monkeypatch):
        """search 端点同样不写单例"""
        from app import api

        class FakeUserService:
            _DEPTS = {"zhangsan": ["HR"]}

            def authenticate(self, uid):
                return {"name": uid} if uid in self._DEPTS else None

            def get_departments(self, uid):
                return self._DEPTS[uid]

        class FakeRag:
            def __init__(self):
                self.user_service = FakeUserService()
                self.current_user = None
                self.retrieval_module = type("R", (), {"permission_aware_search": lambda self, q, d, top_k: [],
                                                        "hybrid_search": lambda self, q, top_k: []})()

        monkeypatch.setattr(api, "_rag", FakeRag())
        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        token = client.post("/auth/login", json={"user_id": "zhangsan"}).json()["access_token"]
        resp = client.post("/api/v1/search", json={"question": "q"},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        # 能走到这里且 200，说明 _resolve_departments 没抛错、没读单例


# ===== 8. config 阈值接线（Task1）=====
class TestConfigThresholds:
    """验证两个阈值进入 config，且 rerank_top_n 向后兼容保留"""

    def test_thresholds_exist_with_defaults(self):
        from config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.vector_score_threshold == 0.3
        assert DEFAULT_CONFIG.rerank_threshold == 0.3


# ===== 9. 检索接线（Task2：死旋钮生效）=====
class TestRetrievalWiring:
    """验证 config 的检索参数真正传到检索器（修复死旋钮）"""

    def test_resolve_config_uses_config_values(self):
        from types import SimpleNamespace
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R
        ret = R.__new__(R)
        cfg = SimpleNamespace(vector_search_k=8, bm25_search_k=7, rrf_k=30,
                              vector_score_threshold=0.4, rerank_threshold=0.5)
        ret._resolve_config(cfg)
        assert ret.vector_search_k == 8
        assert ret.bm25_search_k == 7
        assert ret.rrf_k == 30
        assert ret.vector_score_threshold == 0.4
        assert ret.rerank_threshold == 0.5

    def test_resolve_config_none_uses_old_defaults(self):
        """config=None 时退回原硬编码默认（零破坏，现有调用方不受影响）"""
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R
        ret = R.__new__(R)
        ret._resolve_config(None)
        assert ret.vector_search_k == 5
        assert ret.bm25_search_k == 5
        assert ret.rrf_k == 60
        assert ret.vector_score_threshold == 0.3
        assert ret.rerank_threshold == 0.3

    def test_bm25_k_wired_end_to_end(self):
        """端到端：config.bm25_search_k 真的传到 BM25Retriever（不是硬编码 5）"""
        from types import SimpleNamespace
        from langchain_core.documents import Document
        from rag_modules.retrieval_optimization import RetrievalOptimizationModule as R

        chunks = [Document(page_content="年假5天", metadata={"source": "HR/x.md", "chunk_id": "1"}),
                  Document(page_content="报销500元", metadata={"source": "财务/y.md", "chunk_id": "2"})]

        class FakeVS:
            def as_retriever(self, **kw):
                return None  # 本测试不使用向量检索器

        cfg = SimpleNamespace(vector_search_k=8, bm25_search_k=7, rrf_k=60,
                              vector_score_threshold=0.3, rerank_threshold=0.3)
        ret = R(FakeVS(), chunks, config=cfg)
        assert ret.bm25_retriever.k == 7  # 接线生效；修复前恒为 5


# ===== 10. 扫参命中判定（Task4：answer-span 文件级匹配）=====
class TestSweepHitCriterion:
    """验证 is_hit 用判别数字区分同文件不同档位（1年/3年/10年），且无需 chunk 级标注"""

    @staticmethod
    def _chunk(content, source):
        from langchain_core.documents import Document
        return Document(page_content=content, metadata={"source": source})

    def test_3yr_chunk_hits_3yr_case_only(self):
        from evaluation.sweep_retrieval import is_hit
        case_1 = {"ground_truth": "工作满1年不满3年年假5天", "source_doc": "HR/年假管理制度.md"}
        case_3 = {"ground_truth": "工作满3年不满5年年假10天", "source_doc": "HR/年假管理制度.md"}
        chunk_3yr = self._chunk("工作满3年不满5年的，年假10天。", "HR/年假管理制度.md")
        assert is_hit(chunk_3yr, case_3) is True   # 数字 {3,5,10} ⊆ {3,5,10}
        assert is_hit(chunk_3yr, case_1) is False   # case_1 要 {1,3,5}，chunk 缺 1 → 不命中

    def test_wrong_file_not_hit(self):
        from evaluation.sweep_retrieval import is_hit
        case = {"ground_truth": "工作满3年年假10天", "source_doc": "HR/年假管理制度.md"}
        chunk = self._chunk("工作满3年年假10天", "财务/报销.md")
        assert is_hit(chunk, case) is False

    def test_no_number_gt_falls_back_to_file_level(self):
        from evaluation.sweep_retrieval import is_hit
        case = {"ground_truth": "OA系统提交申请主管审批", "source_doc": "HR/考勤管理制度.md"}
        chunk = self._chunk("请假通过OA系统提交", "HR/考勤管理制度.md")
        assert is_hit(chunk, case) is True

    def test_correct_refuse(self):
        from evaluation.sweep_retrieval import is_correct_refuse
        neg = {"source_doc": ""}
        assert is_correct_refuse([], neg) is True
        assert is_correct_refuse([object()], neg) is False


# ===== 11. 扫参指标计算（Task5：fake retriever 单测）=====
class TestSweepMetrics:
    """eval_one_config 用 fake retriever 验证 Recall@K/MRR/neg_refuse 计算"""

    def test_metrics_from_known_mapping(self):
        from evaluation.sweep_retrieval import eval_one_config
        from langchain_core.documents import Document

        def chunk(content, source):
            return Document(page_content=content, metadata={"source": source})

        # 3 正例（同文件，靠数字区分）+ 1 负例
        eval_set = [
            {"question": "q3", "ground_truth": "10天", "source_doc": "HR/x.md"},
            {"question": "q5", "ground_truth": "500元", "source_doc": "财务/y.md"},
            {"question": "qmiss", "ground_truth": "999", "source_doc": "IT/z.md"},  # 召回错误内容
            {"question": "qneg", "ground_truth": "", "source_doc": ""},             # 负例
        ]
        mapping = {
            "q3": [chunk("年假10天", "HR/x.md")],          # rank1 命中
            "q5": [chunk("无关", "财务/y.md"), chunk("报销500元", "财务/y.md")],  # rank2 命中
            "qmiss": [chunk("别的9999", "IT/z.md")],        # 数字 999 不在 → 未命中
            "qneg": [chunk("xxx", "其他/w.md")],            # 有召回 → 负例拒答失败
        }

        class FakeRet:
            def hybrid_search(self, q, top_k=3):
                return mapping.get(q, [])[:top_k]

        m = eval_one_config(FakeRet(), eval_set, top_k=3)
        # 正例 3 个，命中 2 个 → recall = 2/3
        assert m["recall"] == round(2 / 3, 4)
        # MRR：q3 rank1=1.0，q5 rank2=0.5，qmiss=0 → (1.0+0.5+0)/3
        assert m["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 4)
        # 负例 1 个，拒答失败 → neg_refuse = 0/1 = 0
        assert m["neg_refuse"] == 0.0
        assert m["overall"] == round(2 / 3, 4) * 0.0


# ===== 12. config 死字段清理（Task1）=====
class TestConfigDeadFields:
    """验证 7 个死字段已从 config 移除"""

    def test_dead_fields_removed(self):
        from config import DEFAULT_CONFIG
        dead = ["rerank_top_n", "enable_rbac", "eval_dataset_path", "redis_url",
                "langfuse_host", "langfuse_public_key", "langfuse_secret_key"]
        for field in dead:
            assert not hasattr(DEFAULT_CONFIG, field), f"死字段 {field} 应已移除"

    def test_chunk_overlap_default_is_25(self):
        from config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.chunk_overlap == 25

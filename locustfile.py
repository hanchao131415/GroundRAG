"""
Locust 压测脚本（review Q1：能扛多少 QPS？）

用法：
  # 1. 先启动 RAG 服务
  uvicorn app.api:app --port 8000

  # 2. 启动 locust（Web UI 模式）
  locust -f locustfile.py
  # 打开 http://localhost:8089 配置并发数

  # 3. 或纯命令行模式（无 UI，适合 CI / 脚本）
  locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8000

压测策略：
  - HealthUser：只打 /health（测基础设施瓶颈，不碰 LLM）
  - SearchUser：打 /api/v1/search（碰检索 + Reranker，不碰 LLM 生成，省 token）
  - ChatUser：打 /api/v1/chat（全链路，最真实，但烧 token，建议少量并发）

review 关心的数据：
  - /health 的 RPS（应该 >1000，纯内存返回）
  - /api/v1/search 的 P95 延迟（<2s 合格，含 FAISS + BM25 + Reranker）
  - /api/v1/chat 的 P95 延迟（<10s 合格，含 LLM 生成）
"""

import os
from locust import HttpUser, task, between


# 测试用户（需与 data/users.json 中的用户一致）
TEST_USER = os.getenv("LOCUST_TEST_USER", "zhangsan")


class HealthUser(HttpUser):
    """轻量用户：只打健康检查（测基础设施吞吐）"""
    weight = 3  # 占比最高
    wait_time = between(0.1, 0.5)

    @task
    def health(self):
        self.client.get("/health", name="/health")


class SearchUser(HttpUser):
    """检索用户：打 /api/v1/search（测检索链路，不烧 LLM token）"""
    weight = 2
    wait_time = between(1, 3)

    def on_start(self):
        """登录拿 JWT token"""
        resp = self.client.post("/auth/login", json={"user_id": TEST_USER})
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task
    def search(self):
        if not self.token:
            return
        questions = [
            "年假几天",
            "出差报销上限",
            "密码多久更换",
            "考勤打卡时间",
            "办公用品怎么领",
        ]
        import random
        q = random.choice(questions)
        self.client.post(
            "/api/v1/search",
            json={"question": q, "top_k": 3},
            headers=self.headers,
            name="/api/v1/search",
        )


class ChatUser(HttpUser):
    """全链路用户：打 /api/v1/chat（最真实，但烧 token，低并发）"""
    weight = 1
    wait_time = between(3, 8)  # 模拟用户思考间隔

    def on_start(self):
        resp = self.client.post("/auth/login", json={"user_id": TEST_USER})
        if resp.status_code == 200:
            self.token = resp.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task
    def chat(self):
        if not self.token:
            return
        questions = [
            "工作满1年年假有几天",
            "出差住宿费报销上限是多少",
            "信息安全密码更换周期",
        ]
        import random
        q = random.choice(questions)
        self.client.post(
            "/api/v1/chat",
            json={"question": q, "stream": False},
            headers=self.headers,
            name="/api/v1/chat",
        )

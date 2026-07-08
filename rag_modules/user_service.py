"""
用户与权限模块（RBAC 核心模拟）

真实系统的权限链：
  ① 用户身份（登录态拿 user_id）
  ② 权限映射（user_id → 可见部门，真实从数据库查）
  ③ 检索过滤（用②结果做 permission_aware_search）

本模块模拟①②——用配置文件代替数据库，让"用户→部门"可测、可视。
真实生产时，只需把 _load_users() 换成查数据库即可，检索过滤逻辑不变。

这也是面试可讲的点："权限系统分层设计——身份/映射/过滤解耦，换数据源不影响检索"。
"""

import json  # 标准库：读写 JSON 配置文件（替代数据库）
import logging  # 标准库：日志
from pathlib import Path  # 标准库：跨平台路径处理（比 os.path 更面向对象）
from typing import Dict, List, Optional  # 类型注解：让返回值/参数类型一目了然

# 模块级 logger：统一用 __name__ 命名空间，便于按模块过滤日志
logger = logging.getLogger(__name__)


class UserService:
    """
    用户与权限服务（模拟 RBAC：user_id → 可见部门）。

    本类是 RBAC 权限链中的"映射层"（②），只负责"用户能看哪些部门"：
      ① 身份层：authenticate(user_id) → 用户是谁
      ② 映射层：get_departments(user_id) → 能看哪些部门  ← 本类核心
      ③ 过滤层：permission_aware_search → 用②的结果过滤检索（在别处实现）

    分层解耦卖点：把 _load_users() 换成查数据库，检索过滤逻辑一行不用改。
    """

    def __init__(self, users_file: str = "data/users.json"):
        # 数据源路径：默认读 data/users.json；真实系统这里会换成 DB 连接
        self.users_file = users_file
        # 用户表缓存：{user_id: {name, departments, role}}，启动时一次性加载
        # 类型注解 Dict[str, dict] 表示"键是 user_id(str)，值是用户信息 dict"
        self.users: Dict[str, dict] = {}
        self._load_users()  # 启动即加载用户数据（构造时就初始化，保证对象可用）

    def _load_users(self):
        """
        加载用户配置（真实系统：改成查数据库）。

        策略：文件存在 → 读文件；不存在 → 用内置示例用户并落盘（首启动自举）。
        这样无论是否有配置文件，系统都能跑起来，方便开发/测试。
        """
        p = Path(self.users_file)  # 把字符串路径包装成 Path 对象，支持 .exists()/.read_text()
        if not p.exists():
            # 默认示例用户（首次自动创建，方便测试）
            # departments 里的 "*" 是通配符，表示"能看所有部门"（见 get_departments 处理）
            self.users = {
                "zhangsan": {"name": "张三", "departments": ["HR"], "role": "员工"},
                "lisi":     {"name": "李四", "departments": ["财务"], "role": "员工"},
                "admin":    {"name": "管理员", "departments": ["*"], "role": "管理员"},  # "*" = 全部门可见
            }
            # parents=True 表示连父目录一起创建（如 data/ 不存在则一并建）
            # exist_ok=True 表示目录已存在时不报错（幂等）
            p.parent.mkdir(parents=True, exist_ok=True)
            # 把示例用户写回文件，下次启动直接读文件（ensure_ascii=False 保留中文）
            p.write_text(json.dumps(self.users, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"已创建示例用户配置: {self.users_file}")
        else:
            # 文件已存在：直接读 JSON 还原成 dict（真实系统换成 SQL 查询即可）
            self.users = json.loads(p.read_text(encoding="utf-8"))
            logger.info(f"已加载 {len(self.users)} 个用户")

    def get_departments(self, user_id: str) -> List[str]:
        """
        根据 user_id 获取该用户可见部门。

        这是 RBAC 映射层的对外接口，检索过滤（permission_aware_search）就靠它的返回值。
        真实系统：这里查数据库的 user_role_mapping 表。

        Args:
            user_id: 用户标识（如 "zhangsan"）

        Returns:
            可见部门列表，如 ["HR"]；管理员返回 ["*"]（通配符，表示全部）

        Raises:
            ValueError: 用户不存在时抛错（带提示，方便定位）
        """
        user = self.users.get(user_id)  # dict.get：找不到返回 None（不抛 KeyError）
        if not user:
            # 用户不存在：抛 ValueError 并列出可用用户，便于调试/前端提示
            raise ValueError(f"用户不存在: {user_id}。可用: {list(self.users.keys())}")
        return user["departments"]  # 返回部门列表（admin 的是 ["*"]，由调用方识别通配）

    def list_users(self) -> str:
        """
        列出所有用户（登录时提示用）。

        返回格式化好的多行字符串，可直接打印。uid 用 :12 左对齐占 12 字符，
        让多行用户信息对齐美观。
        """
        lines = []  # 收集每一行，最后 join 成完整字符串
        for uid, info in self.users.items():
            dept = "/".join(info["departments"])  # 多个部门用 "/" 拼接，如 "HR/财务"
            # {uid:12} 表示 uid 左对齐占 12 个字符宽，保证多行对齐
            lines.append(f"  {uid:12} ({info['name']}, {info['role']}, 可见部门: {dept})")
        return "\n".join(lines)  # 用换行连接，返回可直接打印的字符串

    def authenticate(self, user_id: str) -> Optional[dict]:
        """
        模拟登录认证（真实系统：校验密码/JWT）。

        这里是"演示版"——只要 user_id 存在就算登录成功，不校验密码。
        真实生产要换成密码哈希校验或 JWT 验签。

        Args:
            user_id: 登录用户标识

        Returns:
            用户信息 dict（登录成功） 或 None（用户不存在）
        """
        user = self.users.get(user_id)  # 查用户，不存在返回 None
        if user:
            logger.info(f"用户登录: {user_id} ({user['name']})")  # 记录登录日志（审计用）
            return user  # 返回用户信息，供后续生成登录态/做权限判断
        return None  # 用户不存在 → 返回 None，调用方据此判断登录失败

"""
JWT 认证模块（企业级 RBAC 的第一环：身份认证）

为什么用 JWT 而不是 Session：
  - 无状态：服务端不存 session，水平扩容时不需要共享 session 存储
  - 自包含：token 内带 user_id，校验只需密钥，不查库（查库在权限映射那一步）

三层安全模型（对照 user_service.py 的注释）：
  ① 认证（本模块）：JWT 校验 → 拿到可信 user_id（客户端传什么都不信）
  ② 授权（user_service）：user_id → 可见部门
  ③ 过滤（retrieval）：部门 → 检索过滤

设计取舍：
  - 用 PyJWT 而非 python-jose：更轻量，无需 cryptography 加密算法依赖
  - HS256 对称密钥：单服务够用；多服务/微服务才换 RS256（公私钥分离）
  - token 过期时间默认 24h：企业内网系统，不让用户频繁重登

==============================================================================
【核心概念：JWT 到底是什么？】（先看懂这一段，下面的代码就是套模板）
==============================================================================
JWT（JSON Web Token）本质是一个"带防伪签名的字符串"，长得像：
    xxxxx.yyyyy.zzzzz
        ↑     ↑     ↑
      header payload signature

  ① header    —— 头部，JSON 说明"用什么算法签名"，如 {"alg":"HS256","typ":"JWT"}
  ② payload   —— 载荷，JSON 放业务数据，如 {"sub":"u123","exp":1700000000}
                  ⚠ 注意：payload 只是 Base64 编码，不是加密！任何人都能解开看，
                    所以绝对不能在里面放密码/敏感信息。它的"安全"只在于不能被篡改。
  ③ signature —— 签名，用密钥对 "header.payload" 算出的 HMAC，是防伪核心。

校验原理：服务端收到 token 后，用同一个密钥重新算一遍签名，跟 token 里的签名比对：
  - 对不上 → 有人篡改过 payload（比如把 sub 改成 admin）→ 拒绝。
  - 对得上 → payload 可信，可以放心取出 user_id 用。

【HS256 vs RS256，怎么选？】
  - HS256（本模块）：对称加密，签发和校验用【同一个密钥】。
      优点：简单，一个字符串搞定；缺点：任何能校验的方也都能签发，不能把校验权
      安全下放。适合：单体应用、所有服务都信任的内部系统。
  - RS256：非对称加密，私钥签发 + 公钥校验。
      优点：公钥可以到处发（API 网关、前端、第三方），它们只能验不能签；
      缺点：要管公私钥对、依赖稍重。适合：微服务、需要"签发权集中、校验权分散"。

==============================================================================
【#A3 弱点提示：当前 login 接口没有密码校验】（务必知道的安全边界）
==============================================================================
本文件只实现"token 机制"——create_access_token(user_id) 谁来调都能拿 token。
也就是说：只要知道某个 user_id（比如 admin），就能直接换到该用户的合法 token，
这是典型的"身份伪造"风险。完整闭环应是：
    用户提交账号密码 → 查库校验密码哈希 → 校验通过才调 create_access_token。
密码校验逻辑不在本模块（应在 user_service / login 路由里），所以学习本文件时，
请把 create_access_token 想象成"已经登录成功之后"才执行的步骤。生产环境务必补上
密码校验，否则本模块的 JWT 防线形同虚设。
"""

import logging  # 标准库日志：用 __name__ 命名 logger，输出会带上模块名（app.auth），方便定位
import time     # time.time() 返回当前 Unix 时间戳（秒，浮点），JWT 的 iat/exp 都用它
from typing import Optional  # 类型标注：Optional[X] = X | None，表示"可能有值也可能是 None"

import jwt  # 第三方库 PyJWT：JWT 的纯 Python 实现，提供 encode(签发)/decode(校验)
# FastAPI 的 HTTP 相关工具：
#   - Depends：声明依赖注入，FastAPI 会自动调用依赖函数并把返回值注入路由参数
#   - HTTPException：抛出后 FastAPI 直接转成 HTTP 错误响应（带状态码+detail）
#   - Request：当前请求对象，可读写 request.state（跨中间件/路由传数据）
#   - status：HTTP 状态码常量枚举，如 status.HTTP_401_UNAUTHORIZED，避免写裸数字
from fastapi import Depends, HTTPException, Request, status
# HTTPBearer：自动从请求头 Authorization: Bearer <token> 解析出 token 的"提取器"
# HTTPAuthorizationCredentials：HTTPBearer 解析后返回的类型（.credentials 就是 token 字符串）
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# 以模块名建 logger；本文件所有 logger.info/warning 会打上 "app.auth" 前缀
logger = logging.getLogger(__name__)

# Bearer token 提取器（从 Authorization: Bearer <token> 头拿 token）
# auto_error=False：请求头里没有 token 时不让它自动报 403，而是返回 None，
#   把"报不报错"的决定权留给后面的 get_current_user（我们要统一报 401+友好提示）。
security = HTTPBearer(auto_error=False)

# 默认密钥（仅开发用；生产必须通过环境变量 JWT_SECRET 覆盖）
# ⚠ 这个值是公开的，任何看到源码的人都能用它伪造合法 token。
#   生产环境务必在 config.py 里通过 JWT_SECRET 环境变量覆盖成随机长串。
DEFAULT_SECRET = "rag-dev-secret-change-me-in-production"
JWT_ALGORITHM = "HS256"   # 签名算法：HMAC-SHA256（对称），见模块顶部 HS256 vs RS256 说明
JWT_EXPIRE_HOURS = 24      # 默认有效期 24 小时；越短越安全（泄露窗口小），但用户要更频繁重登


def create_access_token(user_id: str, secret: str = "", expires_hours: int = JWT_EXPIRE_HOURS) -> str:
    """
    签发 JWT（登录成功后调用）

    payload 里放什么：
      - sub: user_id（subject，JWT 标准字段）
      - iat: 签发时间（issued at）
      - exp: 过期时间（expiry），到期后 token 自动失效

    参数:
        user_id: 要写入 token 的用户标识（登录成功后从账号系统拿到）
        secret: 签名密钥；留空则用 DEFAULT_SECRET（开发）/ 实际应来自 config 的 JWT_SECRET
        expires_hours: 有效期（小时），默认 24h

    返回:
        形如 "header.payload.signature" 的 JWT 字符串，前端拿到后存 localStorage，
        后续每个请求放在 Authorization: Bearer <token> 头里带上。

    ⚠ #A3 弱点：本函数不做任何身份核验，调用方必须确保 user_id 已通过密码校验！
    """
    # 三元 fallback：调用方没传 secret 就用默认密钥。
    # 实际运行时 get_current_user 那条链会从 config 取真实 JWT_SECRET 传进来，
    # 直接调 create_access_token（如登录路由）若不传 secret，就会落到这个不安全的默认值。
    secret = secret or DEFAULT_SECRET
    now = int(time.time())  # 当前时间戳（秒，整数）—— JWT 标准要求 exp/iat 用整数秒
    # payload（载荷）= 要塞进 token 的业务数据。
    # 字段名 sub/iat/exp 都是 JWT 标准（RFC 7519）定义的"保留 claim"，库会特殊处理：
    payload = {
        "sub": user_id,                  # sub=subject：本 token 代表谁（标准字段，校验后据此取 user_id）
        "iat": now,                      # iat=issued at：签发时刻，便于排查"这个 token 多旧了"
        "exp": now + expires_hours * 3600,  # exp=expiry：过期时刻=现在+有效期。
                                          #   jwt.decode 默认会校验 exp，过了此刻就抛 ExpiredSignatureError。
                                          #   注意：exp 是"绝对时间点"，不是"还有多久"。
    }
    # jwt.encode：用 secret + HS256 对 header.payload 算 HMAC 签名，拼成三段式字符串返回。
    #   它本身不存任何状态，纯计算，所以同一个 payload 每次签出来的 token 字符串完全一样。
    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    logger.info(f"签发 JWT: user={user_id}, 有效期 {expires_hours}h")  # 审计日志：谁、何时、签了多久
    return token


def verify_token(token: str, secret: str = "") -> str:
    """
    校验 JWT，返回 user_id（校验失败抛 401）

    三重校验：
      ① 签名是否正确（防伪造）
      ② 是否过期（exp 字段，jwt 库自动校验）
      ③ sub 字段是否存在（防空 payload）

    参数:
        token: 待校验的 JWT 字符串（来自请求头）
        secret: 校验用的密钥；必须与签发时用的是【同一个】，否则签名对不上 → InvalidTokenError

    返回:
        校验通过则返回 payload 里的 user_id（可信来源，后续鉴权/过滤都以此为准）

    抛出:
        HTTPException(401)：token 过期、无效、或缺少用户标识时
    """
    secret = secret or DEFAULT_SECRET  # 同 create_access_token，取真实密钥或默认值
    try:
        # jwt.decode 是核心：一步完成"验签 + 验过期 + 解析 payload"。
        #   - algorithms 必须传列表（防"alg=none"降级攻击：明确只接受 HS256）。
        #   - 默认会校验 exp，过期就抛 ExpiredSignatureError（见下方分支）。
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        # 过期单独处理：给用户更明确的提示"重新登录"，而不是笼统的"无效"。
        # 这就是上面 create_access_token 里 exp 字段发挥作用的地方——库自动比对当前时间。
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
        )
    except jwt.InvalidTokenError as e:
        # 兜底所有其他错误：签名不对、格式错、被篡改……都归为"无效"。
        # jwt.InvalidTokenError 是 ExpiredSignatureError 的父类，但因上面已先匹配了子类，
        #   这里只会接住"非过期"类的错误。把原始异常 e 带进 detail 便于排查。
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效: {e}",
        )

    # 走到这里说明签名+过期都过了，payload 可信。再取业务字段 sub。
    user_id = payload.get("sub")
    if not user_id:
        # 防御性检查：万一有人用一个合法密钥签了个空 payload（sub 为空），
        #   不能放行，否则后续拿到空 user_id 会出各种诡异问题。
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 缺少用户标识",
        )
    return user_id  # 返回可信 user_id，调用方（路由）据此做后续授权


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    FastAPI 依赖项：从请求中提取并校验 JWT，返回可信的 user_id

    用法（在路由上声明依赖）：
        @app.post("/api/v1/chat")
        async def chat(req: ChatRequest, user_id: str = Depends(get_current_user)):

    关键：user_id 来自 JWT 解码，不是来自请求体 → 客户端伪造 user_id=admin 无效

    依赖链说明:
        Depends(security) → FastAPI 自动执行 HTTPBearer，从请求头
        "Authorization: Bearer <token>" 解析出 credentials（无头则 None）；
        本函数再把这个 credentials 里的 token 交给 verify_token 校验。

    返回:
        可信 user_id。同时把它挂到 request.state，路由内可用 request.state.authenticated_user 取。
    """
    # 延迟导入 config：避免循环依赖（config 不应反向 import auth），
    #   也避免模块加载时就触发配置读取。getattr 取 jwt_secret，取不到则兜底 DEFAULT_SECRET。
    from config import DEFAULT_CONFIG
    secret = getattr(DEFAULT_CONFIG, "jwt_secret", "") or DEFAULT_SECRET

    # 没带 token 的情况：HTTPBearer(auto_error=False) 时 credentials 为 None，
    #   或带了但 .credentials（token 字符串）为空，都算"未提供"。
    #   headers 里加 WWW-Authenticate: Bearer，符合 RFC 6750，告诉客户端"用 Bearer 方式认证"。
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证 Token，请先登录获取 Token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 核心一步：校验 token。verify_token 内部完成验签+验过期，失败直接抛 401（向上冒泡，
    #   FastAPI 自动转成错误响应，路由函数体根本不会执行）。
    user_id = verify_token(credentials.credentials, secret)
    # 把可信 user_id 挂到 request.state，路由内可取（双重保险）
    # —— 即使路由签名里没用 Depends(get_current_user)，只要前置中间件挂过，也能从 state 拿到。
    request.state.authenticated_user = user_id
    return user_id

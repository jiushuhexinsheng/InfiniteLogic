"""
Web 鉴权与限流 / Web authentication + rate limiting.

组件 / Components:
    AuthManager   — API Key 加载与验证 / API Key loading + validation
    RateLimiter   — 滑动窗口限流器（内存）/ Sliding window rate limiter (in-memory)
    auth_middleware — ASGI 中间件 / ASGI middleware

配置驱动 / Config-driven:
    AUTH_ENABLED=true        全局开关 / Global on/off
    API_KEYS=key1,key2       逗号分隔的 API Key / Comma-separated keys
    RATE_LIMIT_PER_MINUTE=20 每分钟限额 / Per-minute cap
    AUTH_SKIP_METRICS=true   /metrics 免鉴权 / Skip auth for /metrics

限流策略 / Rate limiting strategy:
    固定窗口（每分钟 reset），per-key 独立计数。
    Fixed-window per minute, per-key independent counters.
    超限返回 429 + Retry-After + X-RateLimit-* 头。
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config import settings
from src.logging_setup import logger

# ─────────────────────────────────────────────────────────────────────
# 数据模型 / Data models
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ApiKey:
    """API Key 元数据 / API Key metadata."""
    key_hash: str           # SHA256 哈希 / SHA256 hash
    permissions: set[str] = field(default_factory=lambda: {"chat"})
    rate_limit: int = 20    # 每分钟 / per minute


# ─────────────────────────────────────────────────────────────────────
# AuthManager / API Key manager
# ─────────────────────────────────────────────────────────────────────

class AuthManager:
    """
    API Key 管理器 / API Key manager.

    从配置加载 keys，验证 bearer token。
    Loads keys from config, validates bearer tokens.

    安全性 / Security:
        内存中存 SHA256 哈希而非明文。
        Stores SHA256 hashes in memory, not plaintext.
        配置文件仍可存明文（首次加载时哈希化）。
        Config file may still contain plaintext (hashed on load).
    """

    def __init__(self) -> None:
        self._keys: dict[str, ApiKey] = {}  # hash → ApiKey
        self._load_keys()

    def _hash(self, key: str) -> str:
        """SHA256 哈希 / SHA256 hash."""
        return hashlib.sha256(key.strip().encode()).hexdigest()

    def _load_keys(self) -> None:
        """加载配置中的 API Keys / Load keys from config."""
        raw = settings.api_keys
        if not raw:
            logger.info("Auth: no API keys configured")
            return
        for key in raw.split(","):
            key = key.strip()
            if not key:
                continue
            h = self._hash(key)
            self._keys[h] = ApiKey(key_hash=h)
        logger.info("Auth: loaded {} API key(s)", len(self._keys))

    def validate(self, bearer_token: str | None) -> ApiKey | None:
        """
        验证 Bearer token / Validate a bearer token.

        Returns ApiKey if valid, None otherwise.
        """
        if not bearer_token:
            return None
        # 去掉 "Bearer " 前缀（大小写容忍）/ Strip "Bearer " prefix (case-tolerant).
        token = bearer_token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            return None
        h = self._hash(token)
        return self._keys.get(h)

    @property
    def has_keys(self) -> bool:
        """是否有配置任何 key / Whether any keys are configured."""
        return len(self._keys) > 0


# ─────────────────────────────────────────────────────────────────────
# RateLimiter / 限流器
# ─────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    固定窗口限流器（内存实现）/ Fixed-window rate limiter (in-memory).

    适合单进程部署；多副本需换 Redis。
    Suitable for single-process; swap to Redis for multi-replica.

    每个 key_hash 维护独立的 (window_start, count) 对。
    Each key_hash has an independent (window_start, count) pair.
    """

    def __init__(self, max_per_minute: int = 20) -> None:
        self._max = max_per_minute
        # key_hash → [window_start, count]
        self._buckets: dict[str, list[float, int]] = {}
        # 定期清理过期 bucket（每 5 分钟）/ Periodic cleanup every 5 min.
        self._last_cleanup = time.monotonic()

    def _cleanup(self) -> None:
        """清理过期窗口 / Purge expired windows."""
        now = time.monotonic()
        if now - self._last_cleanup < 300:  # 5 分钟一次
            return
        self._last_cleanup = now
        expired = [
            h for h, (ws, _) in self._buckets.items()
            if now - ws >= 60
        ]
        for h in expired:
            del self._buckets[h]

    def check(self, key_hash: str) -> tuple[bool, int]:
        """
        检查是否允许此次请求 / Check if this request is allowed.

        Returns:
            (allowed, remaining) — allowed 为 True 表示放行。
        """
        self._cleanup()
        now = time.monotonic()
        if key_hash not in self._buckets:
            self._buckets[key_hash] = [now, 0]

        ws, count = self._buckets[key_hash]
        # 窗口过期 → 重置 / Window expired → reset.
        if now - ws >= 60:
            self._buckets[key_hash] = [now, 0]
            count = 0

        if count >= self._max:
            remaining_seconds = int(60 - (now - ws))
            return False, max(0, remaining_seconds)

        self._buckets[key_hash][1] = count + 1
        remaining = self._max - (count + 1)
        return True, remaining


# ─────────────────────────────────────────────────────────────────────
# 无需鉴权的路径 / Paths exempt from auth
# ─────────────────────────────────────────────────────────────────────

# 默认公开路径：根页面 + 健康检查 + Prometheus
_PUBLIC_PATHS: set[str] = {"/", "/metrics"}
if settings.auth_skip_metrics:
    _PUBLIC_PATHS.add("/metrics")


def _is_public(path: str) -> bool:
    """判断路径是否无需鉴权 / Check if a path is exempt from auth."""
    # 精确匹配 / Exact match.
    if path in _PUBLIC_PATHS:
        return True
    # 根路径可能带查询参数 / Root path with query string.
    if path.startswith("/?"):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# 全局单例 / Global singletons
# ─────────────────────────────────────────────────────────────────────

_auth_manager: AuthManager | None = None
_rate_limiter: RateLimiter | None = None


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_per_minute=settings.rate_limit_per_minute)
    return _rate_limiter


# ─────────────────────────────────────────────────────────────────────
# ASGI 中间件 / ASGI middleware
# ─────────────────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    ASGI 鉴权 + 限流中间件 / ASGI auth + rate-limit middleware.

    在请求到达路由前拦截，执行鉴权和限流检查。
    Intercepts requests before routing; checks auth + rate limit.

    行为 / Behavior:
        auth_enabled=false → 全部放行（跳过中间件逻辑）
        auth_enabled=true:
            - 公开路径 → 放行
            - 无 API Key → 401
            - 无效 API Key → 403
            - 超限 → 429
            - 通过 → 放行
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # 全局关闭鉴权 → 直通 / Auth disabled → passthrough.
        if not settings.auth_enabled:
            return await call_next(request)

        # 公开路径免鉴权 / Public paths exempt.
        if _is_public(request.url.path):
            return await call_next(request)

        # 无 API Keys 配置 → 拒绝所有请求 / No keys configured → deny all.
        auth = get_auth_manager()
        if not auth.has_keys:
            return JSONResponse(
                {"detail": "No API keys configured. Set API_KEYS in .env."},
                status_code=503,
            )

        # 提取 Bearer token / Extract bearer token.
        token = request.headers.get("Authorization")
        api_key = auth.validate(token)
        if api_key is None:
            return JSONResponse(
                {"detail": "Unauthorized. Provide a valid API key via Authorization: Bearer <key>."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 限流检查 / Rate limit check.
        limiter = get_rate_limiter()
        allowed, remaining = limiter.check(api_key.key_hash)
        if not allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Retry later."},
                status_code=429,
                headers={
                    "Retry-After": str(remaining),
                    "X-RateLimit-Limit": str(settings.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(remaining),
                },
            )

        # 注入 rate limit 头到正常响应 / Inject rate limit headers.
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

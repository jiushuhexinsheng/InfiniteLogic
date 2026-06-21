"""
LLM 响应缓存 / LLM response cache.

精确匹配缓存：相同 (model, messages, tools) → 相同响应。
Exact-match cache: same (model, messages, tools) → same response.

设计 / Design:
    - TTL 过期 / TTL-based expiration
    - LRU 淘汰（条目数达上限时）/ LRU eviction when max entries reached
    - 缓存 key = SHA256(model + JSON(messages) + JSON(tools))
    - 缓存 value = 完整 assistant message dict

集成点 / Integration:
    在 LlmClient.retry_stream_chat() 中：
    - 调用前查缓存 → 命中则直接 yield 缓存的 content + done
    - 调用成功后写缓存

为什么不用语义缓存 / Why not semantic cache:
    语义缓存（embedding 相似度）开销大（需额外 embedding 调用），
    且对 ReAct agent 的 tool-call 场景命中率不高。
    精确匹配零额外推理开销，适合开发调试和重复查询场景。
    Semantic cache adds embedding overhead; exact match has zero extra
    inference cost, suitable for dev/debug and repeated queries.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from src.config import settings


class ResponseCache:
    """
    TTL + LRU 精确匹配缓存。
    TTL + LRU exact-match cache.
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 300) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        # key_hash → (expires_at, message_dict)
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}
        # LRU 追踪：最近访问的 key 排最后 / LRU tracking: most-recently-used at end.
        self._lru: list[str] = []

    @staticmethod
    def _key(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        """生成缓存 key / Generate cache key."""
        payload = json.dumps(
            {"model": settings.llm_model, "messages": messages, "tools": tools or []},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        """
        查缓存 / Look up cached response.

        Returns the full assistant message dict if hit, None otherwise.
        """
        if not settings.cache_enabled:
            return None

        key = self._key(messages, tools)
        entry = self._store.get(key)
        if entry is None:
            return None

        expires_at, message = entry
        if time.monotonic() > expires_at:
            # 过期清理 / Expired — evict.
            del self._store[key]
            self._lru.remove(key)
            return None

        # LRU 刷新：把当前 key 移到列表末尾 / Bump key to end of LRU list.
        self._lru.remove(key)
        self._lru.append(key)
        return message

    def set(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, message: dict[str, Any]) -> None:
        """
        写缓存 / Store a cached response.
        """
        if not settings.cache_enabled:
            return

        key = self._key(messages, tools)

        # 已存在 → 更新 / Already exists → update.
        if key in self._store:
            self._lru.remove(key)
        elif len(self._store) >= self._max:
            # 淘汰最久未用的条目 / Evict least-recently-used.
            oldest = self._lru.pop(0)
            del self._store[oldest]

        self._store[key] = (time.monotonic() + self._ttl, message)
        self._lru.append(key)

    def clear(self) -> None:
        """清空缓存 / Clear all entries."""
        self._store.clear()
        self._lru.clear()

    @property
    def size(self) -> int:
        """当前条目数 / Current entry count."""
        # 清理过期条目再返回 / Purge expired before returning.
        now = time.monotonic()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
            self._lru.remove(k)
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────
# 全局单例 / Global singleton
# ─────────────────────────────────────────────────────────────────────
_cache: ResponseCache | None = None


def get_cache() -> ResponseCache:
    """获取全局缓存单例 / Get global cache singleton."""
    global _cache
    if _cache is None:
        _cache = ResponseCache(
            max_entries=settings.cache_max_entries,
            ttl_seconds=settings.cache_ttl_seconds,
        )
    return _cache


def clear_cache() -> None:
    """清空全局缓存 / Clear global cache."""
    global _cache
    if _cache is not None:
        _cache.clear()

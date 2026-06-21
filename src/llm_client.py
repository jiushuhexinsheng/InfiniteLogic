"""
LLM 客户端封装 / LLM client wrapper.

在 stream_chat 之上提供：
- 连接池复用（Keep-Alive）
- 分级重试策略（429 / 5xx / timeout）
- 熔断器保护（连续失败 → 冷却 → 半开探测）

Layers on top of stream_chat:
- Connection pool (Keep-Alive)
- Tiered retry strategy (429 / 5xx / timeout)
- Circuit breaker (consecutive failures → cooldown → half-open probe)

设计 / Design:
    stream_chat 保持纯粹的 HTTP 流式函数（可独立测试）。
    LlmClient 作为高阶封装，只负责"何时重试、何时熔断"。
    stream_chat stays a pure HTTP streaming function (testable alone).
    LlmClient wraps it with "when to retry, when to break" logic.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.cache import get_cache
from src.config import settings
from src.llm import stream_chat
from src.logging_setup import logger
from src.metrics import LLM_ERRORS_TOTAL
from src.tracing import trace_async_span


# ─────────────────────────────────────────────────────────────────────
# 异常类型 / Exception types
# ─────────────────────────────────────────────────────────────────────
class CircuitBreakerOpenError(Exception):
    """熔断器开启时抛出的异常 / Raised when the circuit breaker is open."""


class RetryExhaustedError(Exception):
    """重试耗尽后抛出的异常 / Raised when all retries are exhausted."""


# ─────────────────────────────────────────────────────────────────────
# 熔断器 / Circuit breaker
# ─────────────────────────────────────────────────────────────────────
# 状态机:
#   CLOSED ──(连续 N 次失败)──▶ OPEN ──(冷却 T 秒)──▶ HALF_OPEN
#     ▲                                                    │
#     └────────(连续 3 次成功)──────────────────────────────┘
#                           │
#                           └──(任意失败)──▶ OPEN
#
# State machine:
#   CLOSED ──(N consecutive failures)──▶ OPEN ──(cooldown T)──▶ HALF_OPEN
#     ▲                                                              │
#     └────────(3 consecutive successes)──────────────────────────────┘
#                                 │
#                                 └──(any failure)──▶ OPEN


class CircuitBreaker:
    """熔断器状态机 / Circuit breaker state machine."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_success_threshold: int = 3,
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._half_open_success = half_open_success_threshold

        self._state: str = "CLOSED"
        self._failure_count: int = 0
        self._success_count: int = 0      # HALF_OPEN 下连续成功计数
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    # ── 查询 / Queries ──────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_open(self) -> bool:
        """调用前检查：熔断器是否拦截请求。Callers check before each request."""
        if self._state != "OPEN":
            return False
        # 冷却期满 → 自动进入 HALF_OPEN / Cooldown expired → HALF_OPEN.
        if time.monotonic() - self._opened_at >= self._cooldown:
            self._state = "HALF_OPEN"
            self._success_count = 0
            logger.info("Circuit breaker: OPEN → HALF_OPEN")
            return False
        return True

    # ── 记录 / Recording ────────────────────────────────────────────

    async def record_success(self) -> None:
        """记录一次成功 / Record a success."""
        async with self._lock:
            if self._state == "HALF_OPEN":
                self._success_count += 1
                if self._success_count >= self._half_open_success:
                    self._state = "CLOSED"
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker: HALF_OPEN → CLOSED")
            else:
                # CLOSED 状态下成功重置失败计数。
                # In CLOSED state, reset failure counter on success.
                self._failure_count = 0

    async def record_failure(self) -> None:
        """记录一次失败 / Record a failure."""
        async with self._lock:
            self._failure_count += 1
            if self._state == "HALF_OPEN":
                # HALF_OPEN 下任意失败 → 回到 OPEN。
                # Any failure in HALF_OPEN → back to OPEN.
                self._state = "OPEN"
                self._opened_at = time.monotonic()
                self._success_count = 0
                logger.warning(
                    "Circuit breaker: HALF_OPEN → OPEN (failure %d)",
                    self._failure_count,
                )
            elif self._state == "CLOSED" and self._failure_count >= self._threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker: CLOSED → OPEN (%d consecutive failures)",
                    self._failure_count,
                )

    async def reset(self) -> None:
        """手动重置熔断器 / Manually reset to CLOSED."""
        async with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._success_count = 0


# ─────────────────────────────────────────────────────────────────────
# 重试决策 / Retry decision logic
# ─────────────────────────────────────────────────────────────────────

# 可重试的 HTTP 状态码 / Retryable HTTP status codes.
_RETRYABLE_STATUS: set[int] = {429, 502, 503, 504}

# 不可重试的 HTTP 状态码（直接抛，不浪费重试次数）。
# Non-retryable status codes.
_PERMANENT_STATUS: set[int] = {400, 401, 402, 403, 404, 422}


def _is_retryable(exc: Exception) -> bool:
    """
    判断异常是否可重试 / Decide if an exception is retryable.

    规则 / Rules:
        httpx.HTTPStatusError with retryable status → True
        httpx.HTTPStatusError with permanent status → False
        httpx.TimeoutException / httpx.ConnectError / httpx.RemoteProtocolError → True
        其他 → False
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in _RETRYABLE_STATUS:
            return True
        if code in _PERMANENT_STATUS:
            return False
        # 未知 4xx 默认不重试，5xx 默认重试 / Unknown 4xx → no, 5xx → yes.
        return code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError,
                         httpx.RemoteProtocolError, httpx.ReadError)):
        return True
    return False


def _retry_after(exc: Exception) -> float | None:
    """
    从 429 响应中提取 Retry-After 秒数。
    Extract Retry-After seconds from a 429 response.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        header = exc.response.headers.get("Retry-After")
        if header is not None:
            try:
                return float(header)
            except ValueError:
                pass
    return None


def _backoff(attempt: int) -> float:
    """
    指数退避 + 随机 jitter。
    Exponential backoff with random jitter.

    base * 2^(attempt-1), capped at backoff_max.
    jitter: ±25% 范围内随机抖动 / random ±25% jitter.
    """
    base = settings.llm_retry_backoff_base
    cap = settings.llm_retry_backoff_max
    raw = base * (2 ** (attempt - 1))
    clamped = min(raw, cap)
    jitter = clamped * 0.25 * (2 * random.random() - 1)  # [-25%, +25%]
    return max(0.0, clamped + jitter)


# ─────────────────────────────────────────────────────────────────────
# LLM 客户端单例 / LLM Client singleton
# ─────────────────────────────────────────────────────────────────────

class LlmClient:
    """
    LLM 客户端封装 / LLM client wrapper.

    用法 / Usage:
        client = LlmClient()
        async for event in client.retry_stream_chat(messages, tools):
            ...

    重用底层 httpx.AsyncClient 实现连接池复用（Keep-Alive）。
    Reuses an underlying httpx.AsyncClient for connection pooling.
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None
        self._breaker = CircuitBreaker(
            failure_threshold=settings.llm_circuit_breaker_threshold,
            cooldown_seconds=settings.llm_circuit_breaker_cooldown,
        )

    async def _get_http(self) -> httpx.AsyncClient:
        """懒加载共享连接池 / Lazy-init shared connection pool."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.llm_request_timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20,
                ),
            )
        return self._http

    async def close(self) -> None:
        """关闭连接池 / Close the connection pool."""
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    async def retry_stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        带重试 + 熔断保护 + 缓存的流式聊天。
        Streaming chat with retry, circuit breaker, and cache.

        行为 / Behavior:
            - 先查缓存 → 命中则直接返回 / Check cache first → return on hit
            - 熔断器 OPEN → CircuitBreakerOpenError（不尝试）
            - 调用 stream_chat：
                · 成功 → record_success → 写缓存 → yield 事件
                · 可重试异常 → 等待退避 → 重试（最多 retry_max 次）
                · 不可重试异常 → 直接抛
                · 重试耗尽 → RetryExhaustedError
            - 每次失败通知熔断器 record_failure
        """
        # ── 缓存查询 / Cache lookup ────────────────────────────
        cache = get_cache()
        cached = cache.get(messages, tools)
        if cached is not None:
            # 缓存命中：模拟流式输出 / Cache hit: simulate streaming output.
            content = cached.get("content", "")
            reasoning = cached.get("reasoning_content", "")
            if reasoning:
                yield {"type": "reasoning_delta", "text": reasoning}
            if content:
                yield {"type": "content_delta", "text": content}
            yield {"type": "done", "message": cached}
            await self._breaker.record_success()
            return

        # ── 熔断检查 / Circuit breaker check ───────────────────
        if self._breaker.is_open:
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN; "
                f"retry after {settings.llm_circuit_breaker_cooldown}s cooldown."
            )

        client = await self._get_http()
        max_retries = settings.llm_retry_max
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):  # 1 次首试 + N 次重试
            try:
                # consume stream_chat 事件直到 done（或中途异常）。
                # Consume stream_chat events until done (or mid-stream exception).
                # 同时捕获 done 事件的 message 用于写缓存。
                # Also capture the done message for caching.
                final_message: dict[str, Any] | None = None
                async with trace_async_span("llm_call", kind="client") as llm_span:
                    if llm_span:
                        llm_span.set_attribute("attempt", attempt + 1)
                        llm_span.set_attribute("model", settings.llm_model)
                        llm_span.set_attribute("msg_count", len(messages))
                    async for event in stream_chat(messages, tools, client=client):
                        yield event
                        if event["type"] == "done":
                            final_message = event.get("message")

                # 走到这里 = done 已 yield; 写缓存 / Cache the result.
                if final_message is not None:
                    cache.set(messages, tools, final_message)
                await self._breaker.record_success()
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

                # 不可重试 → 记录失败 + 直接抛。
                # Non-retryable → record + re-raise immediately.
                if not _is_retryable(exc):
                    LLM_ERRORS_TOTAL.labels(kind=type(exc).__name__).inc()
                    await self._breaker.record_failure()
                    raise

                # 最后一次尝试也失败了 → 记录 + 抛 RetryExhaustedError。
                # Last attempt failed → record + raise RetryExhaustedError.
                if attempt >= max_retries:
                    LLM_ERRORS_TOTAL.labels(kind=type(exc).__name__).inc()
                    await self._breaker.record_failure()
                    raise RetryExhaustedError(
                        f"LLM call failed after {max_retries} retries. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    ) from exc

                # 计算等待时间 / Compute wait time.
                ra = _retry_after(exc)
                if ra is not None:
                    wait = ra
                else:
                    wait = _backoff(attempt + 1)  # attempt 0 是首试，退避从 attempt 1 开始

                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, type(exc).__name__, wait,
                )
                await asyncio.sleep(wait)

        # 理论上不会走到这里（循环内所有路径都 return / raise）。
        # Should never reach here (all paths return or raise).
        raise RetryExhaustedError(f"LLM call failed: {last_exc}")


# ─────────────────────────────────────────────────────────────────────
# 模块级单例 / Module-level singleton
# ─────────────────────────────────────────────────────────────────────
_client: LlmClient | None = None


def get_llm_client() -> LlmClient:
    """获取全局 LlmClient 单例 / Get the global LlmClient singleton."""
    global _client
    if _client is None:
        _client = LlmClient()
    return _client


async def close_llm_client() -> None:
    """关闭全局 LlmClient / Close the global LlmClient."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None

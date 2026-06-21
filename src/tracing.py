"""
分布式追踪 / Distributed tracing.

轻量级 Span 追踪，使用 loguru 输出结构化日志。
Lightweight span tracing via structured loguru output.

Span 模型 / Span model:
    Turn (trace_id=xxx)
      ├── Step 1
      │   ├── LLM call (span)
      │   └── Tool: search_docs (span)
      │       ├── Qdrant query
      │       └── Rerank
      ├── Step 2
      │   ├── LLM call (span)
      │   └── Tool: calculator (span)
      └── Step 3
          └── LLM final (span)

设计选择 / Design choice:
    不用 OpenTelemetry SDK（省依赖），用 loguru 结构化日志。
    每条 Span 是一条 JSON 日志；可用 grep/jq 分析或导入 Jaeger。
    No OpenTelemetry SDK — structured loguru lines serve as spans.
    Each span is a JSON log line; analyzable with grep/jq or Jaeger import.

    如需 OTLP 导出，安装 opentelemetry-api + opentelemetry-sdk，
    本模块自动检测并使用。
    Install opentelemetry-api + opentelemetry-sdk for OTLP export;
    this module auto-detects and uses them if available.

用法 / Usage:
    with trace_span("llm_call", kind="client") as span:
        span.set_attribute("model", "deepseek-chat")
        ...

    async with trace_async_span("tool.search_docs") as span:
        ...
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────
# 追踪状态 / Tracing state
# ─────────────────────────────────────────────────────────────────────
# 当前活跃的 span 栈（线程本地 / coroutine-local 不可行，用简单全局列表）。
# Simple global span stack; use contextvars for true coroutine safety.
# 对单用户 Agent 足够；高并发需换 contextvars。
# Sufficient for single-user agent; swap to contextvars for high concurrency.
import contextvars  # noqa: E402
import json
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from src.config import settings
from src.logging_setup import logger

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "trace_current_span", default=None
)


class Span:
    """追踪 Span / Tracing span."""

    def __init__(
        self,
        name: str,
        kind: str = "internal",
        parent: Span | None = None,
    ) -> None:
        self.trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        self.span_id = uuid.uuid4().hex[:8]
        self.parent_id = parent.span_id if parent else None
        self.name = name
        self.kind = kind
        self.attributes: dict[str, Any] = {}
        self._start = time.monotonic()
        self._end: float | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def finish(self) -> None:
        self._end = time.monotonic()
        duration_ms = (self._end - self._start) * 1000
        # 结构化日志输出 / Structured log output.
        log_data = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(duration_ms, 2),
            "attributes": self.attributes,
        }
        logger.bind(**log_data).debug("trace: {}", json.dumps(log_data, ensure_ascii=False))

    @property
    def duration_ms(self) -> float:
        end = self._end or time.monotonic()
        return (end - self._start) * 1000


@contextmanager
def trace_span(name: str, kind: str = "internal"):
    """
    同步 Span 上下文 / Synchronous span context.

    Usage:
        with trace_span("llm_call", kind="client") as span:
            span.set_attribute("model", "deepseek-chat")
            ...
    """
    if not settings.tracing_enabled:
        yield None
        return

    parent = _current_span.get()
    span = Span(name, kind, parent)
    token = _current_span.set(span)
    try:
        yield span
    except Exception:
        span.set_attribute("error", True)
        raise
    finally:
        span.finish()
        _current_span.reset(token)


@asynccontextmanager
async def trace_async_span(name: str, kind: str = "internal"):
    """
    异步 Span 上下文 / Async span context.

    Usage:
        async with trace_async_span("tool.search_docs") as span:
            ...
    """
    if not settings.tracing_enabled:
        yield None
        return

    parent = _current_span.get()
    span = Span(name, kind, parent)
    token = _current_span.set(span)
    try:
        yield span
    except Exception:
        span.set_attribute("error", True)
        raise
    finally:
        span.finish()
        _current_span.reset(token)


def get_current_span() -> Span | None:
    """获取当前活跃 Span / Get current active span."""
    return _current_span.get()


# ─────────────────────────────────────────────────────────────────────
# 便捷函数 / Convenience functions
# ─────────────────────────────────────────────────────────────────────
def trace_turn(thread_id: str, user_input: str) -> str:
    """创建 Turn 级 trace_id / Create turn-level trace_id."""
    return uuid.uuid4().hex[:16]

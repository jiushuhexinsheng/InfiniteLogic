"""
LLM HTTP 客户端（裸 httpx，无 OpenAI SDK 依赖）
LLM HTTP client using bare httpx — no OpenAI SDK dependency.

为何裸 HTTP / Why bare HTTP:
    OpenAI SDK 帮你处理重试、流式累积、类型化，但同时也"吃掉"了
    一些字段（比如 DeepSeek 的 `reasoning_content`）。直接走 httpx
    我们对 wire-level 数据 100% 可控，DeepSeek thinking 模式无需任何补丁。
    The OpenAI SDK silently drops some fields (e.g. DeepSeek's
    `reasoning_content`). Going to httpx gives us 100% control over the
    wire bytes, so no patches are needed for DeepSeek thinking mode.

核心 API / Core API:
    stream_chat(messages, tools) -> AsyncIterator[dict]
        异步生成器，按 SSE 解析 LLM 流式响应。
        Async generator that parses the LLM SSE stream into events.

事件类型 / Event types yielded:
    - content_delta:   {"type": "content_delta", "text": str}
    - reasoning_delta: {"type": "reasoning_delta", "text": str}
    - tool_call_delta: {"type": "tool_call_delta", "index": int, "id": str|None,
                        "name": str, "arguments": str}
    - done:            {"type": "done", "message": dict}
        message 含完整组装后的 assistant message（含 tool_calls 与 reasoning）
        message holds the fully reassembled assistant message
"""
# from __future__ import annotations 让所有类型注解延迟求值（字符串化），
# 允许在 Python 3.10 上用 list[dict] 这样的语法（PEP 604）。
# Defer evaluation of all type annotations so syntax like list[dict] works
# on Python 3.10 (PEP 604 forward-compat).
from __future__ import annotations

# json: 标准库，用来反序列化 SSE 每行的 data payload。
# Standard library JSON used to decode each SSE line's data payload.
import json
from collections.abc import AsyncIterator

# AsyncIterator: 类型注解；告诉调用者本函数是 async generator。
# Type hint declaring this function as an async generator.
# Optional: 等价于 X | None；保留 Optional 以兼顾老风格阅读。
# Alias for X | None; kept for readability.
from typing import Any

# httpx: 现代异步 HTTP 客户端；支持 streaming + HTTP/2 + 连接池。
# Modern async HTTP client; streaming + HTTP/2 + connection pool.
import httpx

# settings: 全局配置单例，提供 base_url / api_key / model 等。
# Global settings singleton (base_url, api_key, model, ...).
from src.config import settings

# USAGE: 累计 token 与成本的全局单例；本模块在每次响应后写入。
# Global accumulator for token usage + cost; this module writes after each response.
from src.usage import USAGE


def _build_payload(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    stream: bool = True,
    include_usage: bool = True,
) -> dict[str, Any]:
    """
    构造 POST /chat/completions 的请求体。
    Build the body of POST /chat/completions.

    DeepSeek thinking 字段放顶层（与 OpenAI SDK 走 extra_body 不同）。
    DeepSeek's `thinking` flag goes at the top level (no `extra_body` wrapping).
    """
    # payload 是一个普通 dict，httpx 会序列化为 JSON。
    # Plain dict; httpx serializes to JSON automatically.
    payload: dict[str, Any] = {
        "model": settings.llm_model,             # 模型 ID / Model ID
        "messages": messages,                    # 完整消息历史 / Full message history
        "temperature": settings.llm_temperature, # 采样温度 / Sampling temperature
        "max_tokens": settings.llm_max_tokens,   # 单次回复上限 / Per-response cap
        "stream": stream,                        # 是否流式 / Streaming flag
    }
    # 仅在确实有工具时附加 tools 字段，避免空数组干扰某些厂商。
    # Only attach `tools` when present; empty arrays may confuse some providers.
    if tools:
        payload["tools"] = tools
        # tool_choice="auto" 让 LLM 自己决定是否调用工具；可改 "required" 强制。
        # "auto" lets the LLM decide; change to "required" to force a call.
        payload["tool_choice"] = "auto"
    # DeepSeek V4 思考模式开关；顶层 thinking 字段是 DeepSeek 的扩展协议。
    # DeepSeek V4 thinking mode toggle; top-level `thinking` is DeepSeek's extension.
    if settings.llm_thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
        # reasoning_effort: high / max；max 更深思但更慢。
        # reasoning_effort: high (default) or max (deeper but slower).
        payload["reasoning_effort"] = settings.llm_reasoning_effort
    # OpenAI 协议要求流式拿 usage 必须显式启用 stream_options.include_usage。
    # OpenAI protocol: stream + usage requires explicit opt-in.
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _headers() -> dict[str, str]:
    """
    HTTP 请求头 / HTTP request headers.

    Authorization: Bearer <key> 是 OpenAI 协议标准认证方式。
    Accept: text/event-stream 告诉服务器走 SSE 流式响应。
    """
    return {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def _accumulate_tool_calls(
    buffer: dict[int, dict[str, Any]], tc_delta: dict[str, Any]
) -> None:
    """
    把流式 tool_call 片段按 index 累积进 buffer。
    Accumulate streaming tool_call fragments into buffer keyed by index.

    背景 / Background:
        SSE 流中一个 tool_call 会被切成多个 chunk，每个 chunk 含部分字段：
        A tool_call arrives across many chunks; each chunk has partial fields:

            chunk 1: {"index":0, "id":"call_xxx", "function":{"name":"foo"}}
            chunk 2: {"index":0,                  "function":{"arguments":"{\\"a\\""}}
            chunk 3: {"index":0,                  "function":{"arguments":"1}"}}

        我们按 index 分桶，字符串字段累加拼接。
        We bucket by index and concatenate string fields.
    """
    # 当前 chunk 描述哪个 tool_call（index 0/1/2...）；同一轮可能有多个并发 tool_calls。
    # The chunk tells us which tool_call it belongs to (0/1/2...).
    idx = tc_delta["index"]
    # 第一次见到该 index 时初始化空 slot。
    # Initialize an empty slot the first time we see this index.
    if idx not in buffer:
        buffer[idx] = {
            "id": "",                                            # 工具调用 ID / tool call id
            "type": "function",                                  # OpenAI 协议固定 / OpenAI protocol literal
            "function": {"name": "", "arguments": ""},
        }
    slot = buffer[idx]
    # id 通常仅在首个 chunk 中出现一次；覆盖即可。
    # id appears once (usually first chunk); just overwrite.
    if tc_delta.get("id"):
        slot["id"] = tc_delta["id"]
    # function 是嵌套对象；缺失时 fallback 空 dict 避免 AttributeError。
    # function is a nested object; default to {} to avoid AttributeError.
    fn = tc_delta.get("function") or {}
    # 工具名通常一次性给出；保险起见仍走字符串拼接。
    # The name usually arrives in one chunk; still concat for safety.
    if fn.get("name"):
        slot["function"]["name"] += fn["name"]
    # arguments 是 JSON 字符串，按 chunk 拼成完整 JSON。
    # arguments is a JSON string, accreted across chunks.
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


async def stream_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    流式调用 LLM，逐 chunk yield 事件。
    Stream-call the LLM, yielding events per chunk.

    最后 yield 一个 `done` 事件，附带完整组装好的 assistant message：
    Finally yields a `done` event with the fully reassembled assistant message:
        {"type": "done", "message": {
            "role": "assistant",
            "content": str,
            "reasoning_content": str | None,
            "tool_calls": list | None,
        }}

    client 参数：可选共享 httpx.AsyncClient（复用连接池）。
    不传则每次新建；传了则不负责关闭。
    Optional shared AsyncClient for connection pooling.
    Creates one if not provided; caller owns lifecycle if provided.
    """
    # URL = base_url + "/chat/completions"，去掉 base_url 尾部斜杠避免 //。
    # Strip trailing slash on base_url to avoid `//chat/completions`.
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = _build_payload(messages, tools, stream=True)

    # 三个累积缓冲区，对应三类流：内容 / 思考 / 工具调用。
    # Three accumulators for content / reasoning / tool_calls.
    content_buf: list[str] = []
    reasoning_buf: list[str] = []
    tool_calls_buf: dict[int, dict[str, Any]] = {}

    # 有共享 client 直接用；没有就自己建（向后兼容）。
    # Use shared client if provided; create one otherwise (backward compat).
    _own_client: httpx.AsyncClient | None = None
    if client is None:
        _own_client = httpx.AsyncClient(timeout=settings.llm_request_timeout)
        client = _own_client

    try:
        # client.stream(...) 返回一个上下文管理器，进入后 resp 是流式响应。
        # client.stream() returns a context manager; resp inside is a streamed response.
        async with client.stream(
            "POST", url, headers=_headers(), json=payload
        ) as resp:
            # 非 2xx 抛 HTTPStatusError；调用方应该 catch。
            # Raise HTTPStatusError on non-2xx; the caller should catch.
            resp.raise_for_status()

            # aiter_lines() 按 "\n" 拆分；OpenAI SSE 用 "\n\n" 分隔事件，
            # 但每个 data 行就是一条 chunk，按 "data: " 前缀过滤即可。
            # aiter_lines() splits on "\n"; each line prefixed with "data: " is a chunk.
            async for line in resp.aiter_lines():
                # 跳过空行（事件分隔符）和非 data 行（注释、event 等）。
                # Skip blank lines (event separators) and non-data lines.
                if not line or not line.startswith("data: ")  :
                    continue
                # 去掉 "data: " 前缀，得到 JSON 文本。
                # Strip the "data: " prefix, leaving the JSON payload.
                data_str = line[6:]
                # OpenAI 协议在流末发 "[DONE]"，作为客户端终止信号。
                # OpenAI protocol sends "[DONE]" as the terminal marker.
                if data_str == "[DONE]":
                    break
                # 反序列化；少数厂商偶尔发出畸形 chunk，try/except 兜底。
                # Parse JSON; tolerate stray malformed chunks from some providers.
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # 累计 usage 字段（流式最后几个 chunk 通常携带）。
                # Accumulate usage; final chunks usually carry the usage field.
                usage = chunk.get("usage")
                if usage:
                    USAGE.add(usage, settings.llm_model)

                # choices 是数组（OpenAI 协议支持 n>1，但我们只看第 0 个）。
                # `choices` is an array (protocol supports n>1; we only use [0]).
                choices = chunk.get("choices") or []
                if not choices:
                    # 例如纯 usage chunk，没有 choices；继续下一条。
                    # E.g. usage-only chunks: no choices; move on.
                    continue
                # delta 是本次 chunk 的"增量"字段，与 choices[0].message 对应。
                # `delta` is the incremental fragment paralleling choices[0].message.
                delta = choices[0].get("delta") or {}

                # ----- 思考内容（DeepSeek 专有）/ Reasoning content (DeepSeek) -----
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    reasoning_buf.append(reasoning)
                    # 立刻把片段抛给上层 UI 渲染（打字机效果）。
                    # Yield immediately so the UI renders typewriter-style.
                    yield {"type": "reasoning_delta", "text": reasoning}

                # ----- 最终回答片段 / Final answer fragment -----
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_buf.append(content)
                    yield {"type": "content_delta", "text": content}

                # ----- 工具调用片段 / Tool call fragments -----
                tcs = delta.get("tool_calls") or []
                for tc_delta in tcs:
                    # 累积到本地 buffer，用于 done 时输出完整 tool_calls。
                    # Bucket into local buffer for the `done` event.
                    _accumulate_tool_calls(tool_calls_buf, tc_delta)
                    # 同时把片段抛给上层（UI 通常不渲染，agent 不消费这条）。
                    # Also yield raw fragment to the caller (usually unused).
                    yield {
                        "type": "tool_call_delta",
                        "index": tc_delta.get("index", 0),
                        "id": tc_delta.get("id"),
                        "name": (tc_delta.get("function") or {}).get("name") or "",
                        "arguments": (tc_delta.get("function") or {}).get("arguments") or "",
                    }

        # 流结束后组装完整 assistant message。
        # After the stream ends, reassemble the full assistant message.
        message: dict[str, Any] = {
            "role": "assistant",
            # 把所有 content 片段拼成一整段 / Join all content fragments.
            "content": "".join(content_buf),
        }
        # 只有真有思考内容才写字段；空字符串会被下游误读。
        # Only set if non-empty; downstream code checks presence.
        if reasoning_buf:
            message["reasoning_content"] = "".join(reasoning_buf)
        # tool_calls 按 index 升序排列；OpenAI 协议要求保持顺序。
        # Sort tool_calls by index; protocol requires preserved order.
        if tool_calls_buf:
            message["tool_calls"] = [tool_calls_buf[i] for i in sorted(tool_calls_buf)]

        # 终止事件：把完整 message 抛给上层，agent 据此决定下一步。
        # Terminal event: the agent uses this complete message to decide next steps.
        yield {"type": "done", "message": message}

    finally:
        # 如果自己建的 client，关闭它；共享的 client 由调用方管理。
        # Close self-owned client; shared clients are managed by the caller.
        if _own_client is not None:
            await _own_client.aclose()
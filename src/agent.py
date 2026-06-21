"""
ReAct Agent 循环 / ReAct Agent loop.

不依赖 LangChain / LangGraph，纯 while 循环驱动。
Pure while loop driven; no LangChain / LangGraph required.

工作流程 / Flow:
    1. 从 SessionStore 拿历史 / Load history from SessionStore
    2. 拼上 system + 新 user 消息 / Append system + new user message
    3. 调 stream_chat，消费 SSE 事件 / Stream the LLM via stream_chat
    4. 拿到 assistant message：
       Once the assistant message arrives:
         - 无 tool_calls → 终止 / no tool_calls → done
         - 有 tool_calls → 执行 → 追加 ToolMessage → 再调 LLM
           tool_calls → execute → append ToolMessage → loop
    5. 全程通过 AsyncIterator yield UI 事件给 CLI
       Yields UI events to the CLI throughout

事件类型 / Event types yielded to the caller:
    reasoning_delta  — 思考内容片段 / Reasoning fragment
    content_delta    — 回答片段 / Answer fragment
    tool_start       — 工具开始 / Tool started
    tool_end         — 工具结束 / Tool ended
    error            — 错误信息 / Error
    done             — 整轮结束 / Turn finished
"""
# 延迟类型注解求值，兼容 Python 3.10 上的 list[dict] 等语法。
# Defer type annotation evaluation for PEP 604 syntax on Python 3.10.
from __future__ import annotations

# 标准库 / Stdlib only.
import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

# 配置单例：取 recursion_limit / max_history_messages。
# Settings singleton: provides recursion_limit / max_history_messages.
from src.config import settings

# stream_chat: 底层流式调用；用于摘要压缩（一次性调用）。
# stream_chat: low-level streaming call; used for summarization (one-shot).
from src.llm import stream_chat

# LlmClient: 带重试+熔断的高阶封装。
# LlmClient: higher-level wrapper with retry + circuit breaker.
from src.llm_client import CircuitBreakerOpenError, RetryExhaustedError, get_llm_client

# loguru logger 单例 / loguru singleton logger.
from src.logging_setup import logger

# Prometheus 计数器与计时器 / Prometheus counters + timers.
from src.metrics import (
    LLM_ERRORS_TOTAL,
    TOOL_CALLS_TOTAL,
    TTFT_LATENCY,
    TURNS_TOTAL,
    time_tool,
    time_turn,
)

# 提示词模块 / Prompts module.
from src.prompts.agent import (
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
    SUMMARY_PREFIX,
    build_system_prompt,
)
from src.prompts.rules import SAFE_PARALLEL_TOOLS

# 异步会话存储 / Async session store.
from src.session import SessionStore

# 工具注册中心 / Tool registry.
from src.tools import TOOLS

# 分布式追踪 / Distributed tracing.
from src.tracing import trace_async_span


def _trim_history(
    messages: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    """
    历史裁剪 / Trim message history.

    保留首条 system + 最近 N 条；同时尽量不切断 tool_call ↔ tool_message 配对。
    Keep the leading system + last N messages; try not to break
    tool_call ↔ tool_message pairs.

    为什么不能直接 messages[-N:] / Why not simply messages[-N:]:
        若窗口正好把 assistant(tool_call) 切在外面、只留 tool message 在窗口内，
        OpenAI 协议会报错（孤立的 tool message 无父 tool_call）。
        Cutting in the middle of a tool_call/tool_message pair leaves an
        orphan tool message → OpenAI protocol error.
    """
    # 没超过上限就原样返回，零拷贝最快路径。
    # Below the cap → return as-is (fastest path).
    if len(messages) <= max_messages:
        return messages

    # 拆出前导 system 消息 / Split off the leading system message(s).
    system_msgs: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        # 只把"开头连续的 system"视作 prompt 系统消息；中间出现的不动。
        # Only treat leading system msgs as prompt; mid-list ones stay in rest.
        if m.get("role") == "system" and not rest:
            system_msgs.append(m)
        else:
            rest.append(m)

    # 系统消息占用配额，剩下的额度给最近消息。
    # System msgs consume budget; remainder goes to recent ones.
    keep_count = max(1, max_messages - len(system_msgs))
    tail = rest[-keep_count:]

    # 若 tail 第一条是 tool 角色，意味着它失去了父 tool_call assistant 消息。
    # 这种孤儿会被 API 拒绝；从头丢直到遇到非 tool 消息。
    # If tail starts with a tool message, it lost its parent assistant; drop.
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)

    # 系统消息 + 裁剪后的尾部 = 喂给 LLM 的完整 messages。
    # System msgs + trimmed tail = what we feed the LLM.
    return system_msgs + tail


async def _summarize_history(
    history: list[dict[str, Any]],
    session: SessionStore,
    thread_id: str,
) -> list[dict[str, Any]]:
    """
    压缩会话历史 / Summarize conversation history.

    当历史消息超过 agent_summarize_threshold 时：
    1. 取出最旧的部分（system 消息除外）
    2. 调 LLM 生成一段摘要
    3. 替换旧消息为摘要 system 消息
    4. 从 SQLite 删除旧消息

    When history exceeds agent_summarize_threshold:
    1. Take oldest portion (excluding system msgs)
    2. Call LLM to generate a summary
    3. Replace old msgs with summary as system message
    4. Delete old msgs from SQLite
    """
    threshold = settings.agent_summarize_threshold
    keep = settings.agent_summarize_keep_recent

    # 不足阈值不压缩 / Skip if below threshold.
    if len(history) <= threshold:
        return history

    # 分离 system 消息与对话消息 / Split system vs conversation messages.
    system_msgs: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for m in history:
        if m.get("role") == "system" and not rest:
            system_msgs.append(m)
        else:
            rest.append(m)

    # 对话消息也不足阈值 / Conversation msgs also below threshold.
    if len(rest) <= threshold:
        return history

    # 取最旧的 60% 对话消息作为压缩对象 / Take oldest 60% for compression.
    compress_count = max(len(rest) - keep, int(len(rest) * 0.6))
    to_compress = rest[:compress_count]
    to_keep = rest[compress_count:]

    # 拼成纯文本供 LLM 摘要 / Flatten for LLM summarization.
    history_text_parts: list[str] = []
    for m in to_compress:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str) and content:
            history_text_parts.append(f"[{role}]: {content}")
        # 包含 tool 调用信息 / Include tool call info.
        tcs = m.get("tool_calls")
        if tcs:
            for tc in tcs:
                history_text_parts.append(
                    f"[tool_call]: {tc['function']['name']}({tc['function'].get('arguments','')})"
                )

    history_text = "\n".join(history_text_parts)
    if not history_text.strip():
        return system_msgs + to_keep

    # 调 LLM 生成摘要（一次性，无工具，无流式）。
    # One-shot LLM call for summarization (no tools, no streaming).
    summary_prompt = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": SUMMARIZE_USER.format(history_text=history_text)},
    ]
    try:
        assistant_msg: dict[str, Any] | None = None
        async for event in stream_chat(summary_prompt, tools=None):
            if event["type"] == "done":
                assistant_msg = event["message"]
        summary_text = (assistant_msg or {}).get("content", "") if assistant_msg else ""
    except Exception:
        logger.exception("History summarization failed")
        return history  # 失败不丢历史 / Don't lose history on failure.

    if not summary_text.strip():
        return system_msgs + to_keep

    # 从 SQLite 删除旧消息 / Delete old msgs from SQLite.
    try:
        deleted = await session.delete_old_messages(thread_id, keep_recent=keep + 1)
        logger.info(
            "History summarized: {} msgs compressed, {} deleted from DB",
            compress_count, deleted,
        )
    except Exception:
        logger.exception("Failed to delete old messages after summarization")

    # 构造新历史：system 消息 + 摘要 + 保留的最近消息。
    # Rebuild: system msgs + summary + kept recent msgs.
    summary_msg = {
        "role": "system",
        "content": SUMMARY_PREFIX.format(text=summary_text),
    }
    return system_msgs + [summary_msg] + to_keep


async def run_turn(
    user_input: str,
    thread_id: str,
    session: SessionStore,
) -> AsyncIterator[dict[str, Any]]:
    """
    跑一轮对话 / Run one conversation turn.

    yield UI 事件给 CLI 实时渲染；最后 yield "done" 事件。
    Yields UI events for the CLI to render in real time; final "done" event
    closes the turn.

    Args:
        user_input: 本轮用户输入文本 / User's text for this turn
        thread_id:  会话 UUID；SessionStore 用它定位历史
                    Conversation UUID used to fetch / store history
        session:    已打开的 SessionStore / Opened SessionStore instance
    """
    # ── 1. 加载历史 + 注入系统提示 + 历史压缩 ────────────────
    # 1. Load history + ensure system prompt + compress if needed.
    history = await session.load_messages(thread_id)
    # 首次或没有 system 的会话：在最前面补上。
    # First-time or missing system: prepend it.
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": build_system_prompt(TOOLS.schemas())})

    # 历史压缩：超阈值时 LLM 摘要 + 删除旧消息。
    # History compression: summarize + delete when exceeding threshold.
    history = await _summarize_history(history, session, thread_id)

    # 把本轮用户输入加入 history 并立即持久化（crash 安全）。
    # Append + immediately persist user msg (crash-safe).
    user_msg = {"role": "user", "content": user_input}
    history.append(user_msg)
    await session.append(thread_id, user_msg)

    # ── 2. 启动 Prometheus turn 计时器 ─────────────────────────
    # 2. Start the Prometheus turn timer.
    #
    # 不能用 `with time_turn():` 包整个 async generator，因为 yield 会让
    # 控制流暂时离开本函数；Python ≤3.10 的 contextmanager 在生成器里行为
    # 复杂。这里手动 __enter__/__exit__，确保 turn 结束（成功 / 错误 /
    # recursion_limit）都能调用 __exit__。
    # Can't use `with` around an async generator: yields hand control back
    # to the caller; ≤3.10 contextmanager + generator interplay is subtle.
    # We invoke __enter__/__exit__ manually to keep timing correct.
    turn_timer = time_turn()
    turn_timer.__enter__()
    # 默认终态是 error；只在正常结束路径上改成 done。
    # Default to error; flip to done only on the success path.
    terminal_status = "error"

    # ── 3. ReAct 主循环 ────────────────────────────────────────
    # 3. Main ReAct loop.
    for step in range(settings.agent_recursion_limit):
        # 每轮喂给 LLM 前裁剪一次历史。
        # Trim history before each LLM call.
        trimmed = _trim_history(history, settings.agent_max_history_messages)

        try:
            assistant_message: dict[str, Any] | None = None
            # async for 消费 retry_stream_chat 产出的事件流。
            # async for consumes the event stream from retry_stream_chat.
            # Retry + circuit breaker are handled inside.
            _ttft_start = time.monotonic()
            _ttft_recorded = False
            async for event in get_llm_client().retry_stream_chat(
                trimmed, tools=TOOLS.schemas()
            ):
                etype = event["type"]
                if etype == "reasoning_delta":
                    # 思考内容片段直接转发给上层。
                    yield {"type": "reasoning_delta", "text": event["text"]}
                elif etype == "content_delta":
                    # 最终回答片段 + TTFT 记录。
                    # Final answer fragment + TTFT recording.
                    if not _ttft_recorded:
                        TTFT_LATENCY.observe(time.monotonic() - _ttft_start)
                        _ttft_recorded = True
                    yield {"type": "content_delta", "text": event["text"]}
                elif etype == "tool_call_delta":
                    # 不向 UI 输出 tool_call 流碎片（噪音太多，UI 等完整后展示）。
                    # Skip per-chunk tool_call fragments to avoid UI noise.
                    pass
                elif etype == "done":
                    # 完整 assistant message；保留下来决定下一步。
                    # Full assistant message; decide next step from this.
                    assistant_message = event["message"]
        except CircuitBreakerOpenError:
            # 熔断器开启：快速失败，提示用户稍后重试。
            # Circuit breaker open: fail fast, tell user to retry later.
            logger.warning("Circuit breaker open at step {}", step)
            yield {
                "type": "error",
                "message": (
                    "Service temporarily unavailable. "
                    "The LLM provider is experiencing issues. "
                    f"Please retry in {settings.llm_circuit_breaker_cooldown}s."
                ),
            }
            TURNS_TOTAL.labels(status="error").inc()
            turn_timer.__exit__(None, None, None)
            return
        except RetryExhaustedError as exc:
            # 重试耗尽：临时故障持续，给用户清晰信息。
            # Retries exhausted: transient failure persisted. Give clear info.
            logger.exception("LLM retries exhausted at step {}", step)
            yield {"type": "error", "message": f"LLM call failed: {exc}"}
            TURNS_TOTAL.labels(status="error").inc()
            turn_timer.__exit__(None, None, None)
            return
        except Exception as exc:  # noqa: BLE001
            # 不可重试错误（4xx 认证/参数错等）→ 立即终止。
            # Non-retryable errors (4xx auth/params) → terminate immediately.
            logger.exception("LLM call failed at step {}", step)
            LLM_ERRORS_TOTAL.labels(kind=type(exc).__name__).inc()
            yield {"type": "error", "message": f"LLM call failed: {exc}"}
            TURNS_TOTAL.labels(status="error").inc()
            turn_timer.__exit__(None, None, None)
            return

        # 极少数情况下 stream_chat 异常未抛但 message 为空：自保。
        # Rare: stream_chat finished without exception but no message.
        if assistant_message is None:
            yield {"type": "error", "message": "Empty assistant message"}
            TURNS_TOTAL.labels(status="error").inc()
            turn_timer.__exit__(None, None, None)
            return

        # 把完整 assistant 消息加入 history 并持久化。
        # Append + persist the assistant message.
        history.append(assistant_message)
        # 收集本轮需批量持久化的消息（assistant + tool msgs）。
        # Collect messages for batch persist (assistant + tool msgs).
        pending_db: list[dict[str, Any]] = [assistant_message]

        # 取出 tool_calls；空列表 == None 都视为"无工具调用"。
        # Extract tool_calls; treat empty / None as "no tool calls".
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            # 终止条件：LLM 输出最终答案，未再请求工具。
            # Termination: LLM produced final answer, no further tools.
            await session.append_many(thread_id, pending_db)
            terminal_status = "done"
            yield {"type": "done"}
            TURNS_TOTAL.labels(status=terminal_status).inc()
            turn_timer.__exit__(None, None, None)
            return

        # ── 4. 工具执行（支持并行） ────────────────────────────
        # 4. Tool execution (with parallel support).
        #
        # 无副作用工具可并行（asyncio.gather），有副作用工具保持串行。
        # Side-effect-free tools run in parallel (asyncio.gather);
        # side-effect tools stay sequential.

        if settings.agent_parallel_tools:
            # 分组 / Split into parallel vs sequential groups.
            # 使用 prompts.rules 中定义的安全并行白名单。
            # Use safe-parallel whitelist from prompts.rules.
            safe_calls = [tc for tc in tool_calls if tc["function"]["name"] in SAFE_PARALLEL_TOOLS]
            mutex_calls = [tc for tc in tool_calls if tc["function"]["name"] not in SAFE_PARALLEL_TOOLS]

            # 第一步：并行执行安全工具 / Phase 1: parallel safe tools.
            if safe_calls:
                # 先 yield 全部 tool_start 事件（保持 UI 顺序）。
                # Yield all tool_start events first (preserve UI order).
                parsed_safe: list[tuple[dict, str, dict]] = []
                for tc in safe_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"] or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    parsed_safe.append((tc, name, args))
                    logger.info(
                        "tool_call thread={} step={} name={} args={}",
                        thread_id[:8], step, name, args,
                    )
                    yield {"type": "tool_start", "name": name, "args": args}

                # asyncio.gather 并行执行 / Execute in parallel.
                async def _exec_one(name: str, args: dict) -> tuple[str, str]:
                    """执行单个工具并返回 (tool_status, result)。"""
                    with time_tool(name):
                        result = await TOOLS.acall(name, args)
                    tool_status = "error" if result.startswith("Error") else "ok"
                    return tool_status, result

                parallel_results = await asyncio.gather(
                    *[_exec_one(name, args) for _, name, args in parsed_safe],
                    return_exceptions=True,
                )

                # 按原始顺序处理结果 / Process results in original order.
                for (tc, name, _args), presult in zip(parsed_safe, parallel_results, strict=True):
                    if isinstance(presult, Exception):
                        tool_status = "error"
                        result = f"Error in {name}: {presult}"
                    else:
                        tool_status, result = presult

                    TOOL_CALLS_TOTAL.labels(name=name, status=tool_status).inc()
                    logger.info(
                        "tool_result thread={} name={} status={} output_len={}",
                        thread_id[:8], name, tool_status, len(result),
                    )
                    yield {"type": "tool_end", "name": name, "output": result}

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                    history.append(tool_msg)
                    pending_db.append(tool_msg)

            # 第二步：串行执行有副作用工具 / Phase 2: sequential mutex tools.
            tool_calls = mutex_calls

        # ── 串行执行（有副作用工具 或 并行关闭时全部走这里）───
        # Sequential execution (side-effect tools or when parallel is off).
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            logger.info(
                "tool_call thread={} step={} name={} args={}",
                thread_id[:8], step, name, args,
            )
            yield {"type": "tool_start", "name": name, "args": args}
            async with trace_async_span(f"tool.{name}") as tool_span:
                if tool_span:
                    tool_span.set_attribute("step", step)
                with time_tool(name):
                    result = await TOOLS.acall(name, args)
            tool_status = "error" if result.startswith("Error") else "ok"
            TOOL_CALLS_TOTAL.labels(name=name, status=tool_status).inc()
            logger.info(
                "tool_result thread={} name={} status={} output_len={}",
                thread_id[:8], name, tool_status, len(result),
            )
            yield {"type": "tool_end", "name": name, "output": result}

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            }
            history.append(tool_msg)
            pending_db.append(tool_msg)

        # 本轮所有消息批量持久化 / Batch persist all messages for this step.
        await session.append_many(thread_id, pending_db)

        # 工具执行完，回到 for 顶端再次调用 LLM 决定下一步。
        # After all tools run, loop back to call LLM with updated context.

    # 走到这里表示 for 跑满 recursion_limit 仍未终止。
    # Reaching here means recursion_limit exhausted without a final answer.
    TURNS_TOTAL.labels(status="error").inc()
    turn_timer.__exit__(None, None, None)
    yield {
        "type": "error",
        "message": (
            f"Recursion limit ({settings.agent_recursion_limit}) reached. "
            "Increase AGENT_RECURSION_LIMIT or check for tool loops."
        ),
    }
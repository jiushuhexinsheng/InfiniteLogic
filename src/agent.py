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
import json
from typing import Any, AsyncIterator

# 配置单例：取 recursion_limit / max_history_messages。
# Settings singleton: provides recursion_limit / max_history_messages.
from src.config import settings

# stream_chat: 流式 LLM 调用入口；返回 AsyncIterator[dict] 事件流。
# stream_chat: streaming LLM entry point.
from src.llm import stream_chat

# loguru logger 单例 / loguru singleton logger.
from src.logging_setup import logger

# Prometheus 计数器与计时器 / Prometheus counters + timers.
from src.metrics import (
    LLM_ERRORS_TOTAL,
    TOOL_CALLS_TOTAL,
    TURNS_TOTAL,
    time_tool,
    time_turn,
)

# 异步会话存储 / Async session store.
from src.session import SessionStore

# 工具注册中心 / Tool registry.
from src.tools import TOOLS


# ─────────────────────────────────────────────────────────────────────
# 系统提示 / System prompt
#
# 这段文本会作为 messages[0] 注入到每次 LLM 调用，定义 Agent 的"角色描述"。
# Prepended to every LLM call as messages[0]; defines the Agent's persona.
#
# 重点 / Key points:
#   - 列出所有可用工具供 LLM 决策 / Enumerate tools so the LLM can plan
#   - 显式约束行为（不要写包装脚本、不要重复调用同工具同参）
#     Explicit behavioral constraints (no wrapper scripts, no repeat calls)
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant with access to multiple tools.

Available tools:
- search_docs: Search the LOCAL knowledge base. Use this FIRST for domain-specific questions.
- web_search / web_search_results: Search the internet for current information.
- calculator: Evaluate a math expression.
- get_current_datetime: Get current date/time in any timezone.
- read_file / write_file / list_directory: Interact with files in the workspace.
- run_python_file: Execute a Python file located in the workspace.
- exec_python_snippet: Execute an ad-hoc Python code snippet.

Guidelines:
- Think step-by-step before acting.
- Use tools when external data or computation is needed.
- Be concise but complete.
- If a tool fails, explain why and try an alternative approach.
- To run existing Python code in the workspace, call run_python_file directly.
  Do NOT write wrapper scripts that exec/import the target.
- For ad-hoc Python (quick math, prototyping), call exec_python_snippet.
  Do NOT create permanent files for one-off snippets.
- Never call the same tool with identical arguments twice in a row without
  new information; report the issue to the user instead.
"""


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
    # ── 1. 加载历史 + 注入系统提示 ──────────────────────────────
    # 1. Load history + ensure system prompt is present.
    history = await session.load_messages(thread_id)
    # 首次或没有 system 的会话：在最前面补上。
    # First-time or missing system: prepend it.
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # 把本轮用户输入加入 history 并持久化（agent 崩了也不丢用户消息）。
    # Append + persist the user msg (so it's never lost if agent crashes).
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
            # async for 消费 stream_chat 产出的事件流。
            # async for consumes the event stream from stream_chat.
            async for event in stream_chat(trimmed, tools=TOOLS.schemas()):
                etype = event["type"]
                if etype == "reasoning_delta":
                    # 思考内容片段直接转发给上层（CLI 决定是否显示）。
                    # Forward reasoning fragment to caller.
                    yield {"type": "reasoning_delta", "text": event["text"]}
                elif etype == "content_delta":
                    # 最终回答片段 / Final answer fragment.
                    yield {"type": "content_delta", "text": event["text"]}
                elif etype == "tool_call_delta":
                    # 不向 UI 输出 tool_call 流碎片（噪音太多，UI 等完整后展示）。
                    # Skip per-chunk tool_call fragments to avoid UI noise.
                    pass
                elif etype == "done":
                    # 完整 assistant message；保留下来决定下一步。
                    # Full assistant message; decide next step from this.
                    assistant_message = event["message"]
        except Exception as exc:  # noqa: BLE001
            # stream_chat 抛出（网络断、4xx、5xx 等）→ 记日志 + 指标 + 优雅返回。
            # stream_chat raised (network / 4xx / 5xx). Log + metric + return.
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

        # 把完整 assistant 消息加入 history 并写库（含可能的 reasoning + tool_calls）。
        # Append + persist the assistant message (with reasoning + tool_calls).
        history.append(assistant_message)
        await session.append(thread_id, assistant_message)

        # 取出 tool_calls；空列表 == None 都视为"无工具调用"。
        # Extract tool_calls; treat empty / None as "no tool calls".
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            # 终止条件：LLM 输出最终答案，未再请求工具。
            # Termination: LLM produced final answer, no further tools.
            terminal_status = "done"
            yield {"type": "done"}
            TURNS_TOTAL.labels(status=terminal_status).inc()
            turn_timer.__exit__(None, None, None)
            return

        # ── 4. 顺序执行所有 tool_calls ────────────────────────
        # 4. Execute every tool_call in order.
        #
        # 当前实现：串行。可改 asyncio.gather 并行（注意 file_write 等有副作用工具需互斥）。
        # Currently sequential. Could be asyncio.gather'd, but side-effect
        # tools (file_write, etc.) need mutual exclusion if parallelized.
        for tc in tool_calls:
            name = tc["function"]["name"]
            # arguments 是 JSON 字符串；LLM 偶尔生成空串。
            # arguments is a JSON string; LLM may produce "".
            raw_args = tc["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # 解析失败用空 dict；工具内部会因缺参报错并被 safe_tool 转字符串。
                # On parse failure: pass {} — tool will report missing param error.
                args = {}

            # 结构化日志：thread_id 截短 8 位便于关联前端显示。
            # Structured log; truncate thread_id for readability.
            logger.info(
                "tool_call thread={} step={} name={} args={}",
                thread_id[:8], step, name, args,
            )
            # 给 UI 一个明确的"工具开始"事件。
            # Emit explicit "tool started" event for UI.
            yield {"type": "tool_start", "name": name, "args": args}
            # time_tool 是同步 context manager，包同步代码即可。
            # time_tool is a sync ctx manager; wraps the async call body.
            with time_tool(name):
                # acall 自动处理同步/异步函数 + 异常兜底转字符串。
                # acall handles sync/async dispatch + exception → string.
                result = await TOOLS.acall(name, args)
            # 工具错误约定：返回串以 "Error" 开头视为失败。
            # Tool error convention: result starting with "Error" → failure.
            tool_status = "error" if result.startswith("Error") else "ok"
            TOOL_CALLS_TOTAL.labels(name=name, status=tool_status).inc()
            logger.info(
                "tool_result thread={} name={} status={} output_len={}",
                thread_id[:8], name, tool_status, len(result),
            )
            # UI 收到结果（含截断预览）/ UI gets result (with preview).
            yield {"type": "tool_end", "name": name, "output": result}

            # 必须把 tool result 作为 role=tool 消息塞回 history，
            # 用 tool_call_id 关联回 assistant 的 tool_call。
            # Echo tool result back as a role=tool message, linked by id.
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            }
            history.append(tool_msg)
            await session.append(thread_id, tool_msg)

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
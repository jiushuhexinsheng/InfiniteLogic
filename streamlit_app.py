"""
Streamlit 备选 UI / Streamlit alternative UI.

启动 / Run:
    streamlit run streamlit_app.py

特点 / Features:
    - 单文件简洁布局 / Single-file minimal layout
    - 流式渲染（思考流 / 工具调用 / 回答）/ Streaming render
    - sidebar 实时显示 token & 成本 / Live token & cost in sidebar
    - 多会话支持（session_state 维护 thread_id）
      Multi-thread via Streamlit session_state

注意 / Note:
    Streamlit 自己跑事件循环；我们用 asyncio.run 在每次提交里跑一轮异步生成器。
    Streamlit owns its own loop; we use asyncio.run per submission for the
    async generator.

为什么有 Streamlit 又有 FastAPI / Why both:
    - Streamlit：5 行写完一个 UI，适合原型 / 内部工具
      Streamlit: 5-line UI; suits prototypes / internal tools
    - FastAPI：可对接外部前端、移动端、Prometheus 抓取
      FastAPI: pluggable for external frontends + Prometheus
"""
from __future__ import annotations

# 标准库 / Stdlib.
import asyncio
import uuid

# Streamlit 主体 / Streamlit core.
import streamlit as st

# InfiniteLogic 内部模块 / InfiniteLogic internals.
from src.agent import run_turn
from src.logging_setup import setup_logging
from src.session import SessionStore
from src.tools.rag_tool import reload_hybrid, reload_reranker
from src.usage import USAGE

# 初始化日志（Streamlit 重跑时 setup_logging 幂等不重复）。
# Init logging; setup_logging is idempotent across Streamlit reruns.
setup_logging()

# 页面级配置：必须在任何其他 st 调用前 / Page config must be first.
# page_icon 用 emoji 也行 / page_icon accepts emoji too.
st.set_page_config(page_title="InfiniteLogic", page_icon="\U0001f6e0", layout="wide")
st.title("InfiniteLogic Agent")

# ──────────────────────────────────────────────────────────────────
# 会话状态 / Session state
#
# Streamlit 每次输入都会重跑整个脚本；用 st.session_state 保持跨重跑数据。
# Streamlit reruns the whole script on every interaction; session_state
# persists data across reruns.
# ──────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    # 首次进入：生成 thread_id / First entry: generate thread_id.
    st.session_state.thread_id = str(uuid.uuid4())
if "log" not in st.session_state:
    # 消息日志（用于历史回放）/ Message log for replay.
    st.session_state.log: list[dict] = []  # type: ignore[attr-defined]
if "store" not in st.session_state:
    # SessionStore 在 Streamlit 进程内单例。
    # asyncio.run 在每次新协程上下文构造；session_state 缓存避免重复打开。
    # SessionStore is a singleton within the Streamlit process.
    # asyncio.run() creates a new event loop each call; cache to avoid reopens.
    st.session_state.store = asyncio.run(SessionStore.open())

# 取局部引用方便后续代码 / Local handle for terser code below.
store: SessionStore = st.session_state.store

# ──────────────────────────────────────────────────────────────────
# 侧边栏 / Sidebar
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Session")
    # st.code 用等宽字体展示 thread_id 前 8 位 / Monospace display.
    st.code(st.session_state.thread_id[:8], language="text")
    if st.button("New Thread"):
        # 换 thread_id 等于开新会话 / Rotate thread_id = new conversation.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.log = []
        # st.rerun() 强制立刻重跑脚本，UI 同步刷新 / Force immediate rerun.
        st.rerun()
    if st.button("Clear Thread"):
        # 物理删除 SQLite 行 / Hard-delete rows in SQLite.
        asyncio.run(store.clear(st.session_state.thread_id))
        st.session_state.log = []
        st.rerun()

    st.subheader("Retrieval")
    # st.toggle 返回当前 toggle 状态布尔值 / Returns the toggle's bool state.
    hybrid_on = st.toggle("Hybrid (BM25+vector)", value=True)
    reranker_on = st.toggle("Reranker", value=True)
    if st.button("Apply retrieval settings"):
        # 热切换 + toast 提示 / Hot-swap + toast notification.
        st.toast(reload_hybrid(hybrid_on))
        st.toast(reload_reranker(enabled=reranker_on))

    st.subheader("Usage")
    # st.metric 显示大数字 + 标签 / metric widget for key-value display.
    st.metric("Requests", USAGE.requests)
    st.metric("Total tokens", USAGE.total_tokens)
    st.metric("Cost (USD)", f"${USAGE.cost_usd:.4f}")
    if st.button("Reset usage"):
        USAGE.reset()
        st.rerun()

# ──────────────────────────────────────────────────────────────────
# 历史回放 / Replay log
# 重跑脚本时把已有消息渲染出来（Streamlit 不保留 DOM 状态）。
# Replay accumulated messages on each rerun (Streamlit drops DOM).
# ──────────────────────────────────────────────────────────────────
for entry in st.session_state.log:
    # st.chat_message 是聊天气泡容器 / Chat bubble container.
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# ──────────────────────────────────────────────────────────────────
# 输入框 / Input
# st.chat_input 是固定在底部的聊天输入框 / Bottom-anchored chat input.
# ──────────────────────────────────────────────────────────────────
prompt = st.chat_input("Type a message…")
if prompt:
    # 追加 user 消息到日志并渲染 / Append + render user msg.
    st.session_state.log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 用占位符承载流式输出；后续 .markdown(...) 调用会原地更新。
    # Placeholders hold streaming output; later .markdown() updates in place.
    with st.chat_message("assistant"):
        thinking_box = st.empty()
        answer_box = st.empty()
        tool_box = st.empty()

        # 三个 buffer 累积每段文本，每来一个 delta 重新渲染整段。
        # Three buffers; re-render whole section on each delta.
        thinking_buf: list[str] = []
        answer_buf: list[str] = []
        tool_lines: list[str] = []

        async def _drive() -> None:
            """异步驱动 agent，事件转 Streamlit 渲染。"""
            async for event in run_turn(
                prompt, st.session_state.thread_id, store
            ):
                etype = event["type"]
                if etype == "reasoning_delta":
                    thinking_buf.append(event["text"])
                    # 每次有新片段就整段重渲染（Streamlit 占位符特性）。
                    # Re-render the whole section on each new fragment.
                    thinking_box.markdown(
                        "*Thinking:* " + "".join(thinking_buf)
                    )
                elif etype == "content_delta":
                    answer_buf.append(event["text"])
                    answer_box.markdown("".join(answer_buf))
                elif etype == "tool_start":
                    # 用反引号包工具名 + 参数 / Backticks for inline code style.
                    tool_lines.append(
                        f"`{event['name']}` ← {event['args']}"
                    )
                    tool_box.markdown("\n\n".join(tool_lines))
                elif etype == "tool_end":
                    out = event.get("output", "")
                    # 200 字预览避免淹屏 / Cap preview at 200 chars.
                    preview = out[:200] + ("..." if len(out) > 200 else "")
                    tool_lines.append(f"→ {preview}")
                    tool_box.markdown("\n\n".join(tool_lines))
                elif etype == "error":
                    answer_buf.append(f"\n\n**Error:** {event['message']}")
                    answer_box.markdown("".join(answer_buf))

        # 同步等待协程完成；Streamlit 主线程会一直阻塞到 _drive 结束。
        # Block until _drive completes; Streamlit main thread waits.
        asyncio.run(_drive())

        # 把整段最终答案落入历史日志（下次 rerun 时回放）。
        # Persist final answer into the log for next-rerun replay.
        full_answer = "".join(answer_buf) or "_(no answer)_"
        st.session_state.log.append({"role": "assistant", "content": full_answer})
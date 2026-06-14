"""
Web UI（FastAPI + SSE）/ Web UI (FastAPI + SSE).

端点 / Endpoints:
    GET  /                       — 单页 HTML 前端 / Single-page HTML
    POST /chat                   — 触发一轮对话，SSE 流式返回 / Trigger a turn, SSE stream back
    POST /thread/{tid}/clear     — 清空某会话历史 / Clear a thread
    GET  /usage                  — 当前 token 与成本统计 / Cumulative usage
    GET  /metrics                — Prometheus 抓取端点 / Prometheus scrape

启动 / Run:
    python -m src.web   或   python web_server.py

设计选择 / Design choices:
    - 单页 HTML 内联：零前端构建工具链 / Inline HTML: no frontend toolchain
    - SSE 而非 WebSocket：服务器单向推送够用 / SSE: simpler than WS for one-way
    - SessionStore 全局单例：避免每请求重建连接 / Global store to avoid per-request reconnects
"""
from __future__ import annotations

# 标准库 / Stdlib.
import json
from pathlib import Path

# FastAPI 是基于 Starlette + Pydantic 的 ASGI 框架；高性能、类型友好。
# FastAPI is Starlette + Pydantic; high-perf, typed.
from fastapi import FastAPI, Response
# 三种响应类：HTML 文本、流式、JSON / Three response classes.
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
# Prometheus 文本格式生成器 + Content-Type 常量。
# Prometheus exposition format generator + Content-Type constant.
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
# Pydantic 用作请求体校验 / Pydantic for request body validation.
from pydantic import BaseModel

from src.agent import run_turn
from src.config import settings
from src.logging_setup import setup_logging
from src.session import SessionStore
from src.usage import USAGE

# import 时立刻初始化日志（uvicorn 启动前就配好 sink）。
# Init logging on import — sinks ready before uvicorn boots.
setup_logging()
# FastAPI 应用单例 / FastAPI app singleton.
app = FastAPI(title="OpenBase Web UI")

# SessionStore 单例：所有请求共享一个 aiosqlite 连接。
# Global SessionStore: one shared aiosqlite connection across requests.
_store: SessionStore | None = None


@app.on_event("startup")
async def _startup() -> None:
    """
    应用启动钩子 / App startup hook.

    准备 workspace 目录 + 打开 SessionStore。
    Prepare workspace dir and open SessionStore.
    """
    global _store
    Path(settings.workspace_dir).mkdir(parents=True, exist_ok=True)
    _store = await SessionStore.open()


@app.on_event("shutdown")
async def _shutdown() -> None:
    """
    应用关闭钩子 / App shutdown hook.

    优雅关闭 SessionStore；防止 SQLite 文件锁残留。
    Gracefully close SessionStore to avoid stale SQLite locks.
    """
    if _store is not None:
        await _store.close()


# ----------------------------------------------------------------------
# 前端单页 / Single-page HTML
# ----------------------------------------------------------------------
_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>OpenBase</title>
<style>
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
#log { padding: 16px; height: calc(100vh - 110px); overflow-y: auto; white-space: pre-wrap; }
.turn { margin-bottom: 24px; border-bottom: 1px dashed #333; padding-bottom: 16px; }
.role-user { color: #6ad; font-weight: 600; }
.role-assistant { color: #7dc77a; font-weight: 600; }
.tool { color: #d6a64d; margin-top: 6px; }
.tool-out { color: #999; margin-left: 12px; font-size: 0.9em; }
.thinking { color: #b78fd1; font-style: italic; margin-top: 6px; }
.error { color: #e06464; }
form { display: flex; gap: 8px; padding: 12px; background: #161922; border-top: 1px solid #2a2d36; position: sticky; bottom: 0; }
input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #2a2d36; background: #0f1115; color: #e6e6e6; }
button { padding: 10px 18px; border-radius: 6px; border: 0; background: #4a90e2; color: white; cursor: pointer; }
button.secondary { background: #555; }
.bar { padding: 8px 16px; background: #161922; border-bottom: 1px solid #2a2d36; font-size: 0.85em; color: #888; display: flex; gap: 18px; }
</style>
</head>
<body>
<div class="bar">
  <span>Thread: <span id="thread">-</span></span>
  <span>Usage: <span id="usage">-</span></span>
  <span style="margin-left:auto">
    <button class="secondary" onclick="newThread()">New Thread</button>
    <button class="secondary" onclick="clearThread()">Clear</button>
  </span>
</div>
<div id="log"></div>
<form id="form">
  <input id="input" placeholder="Type a message…" autocomplete="off" />
  <button type="submit">Send</button>
</form>
<script>
let threadId = crypto.randomUUID();
document.getElementById('thread').textContent = threadId.slice(0, 8);

const log = document.getElementById('log');
function append(html) { log.insertAdjacentHTML('beforeend', html); log.scrollTop = log.scrollHeight; }

function newThread() {
  threadId = crypto.randomUUID();
  document.getElementById('thread').textContent = threadId.slice(0, 8);
  log.innerHTML = '';
}

async function clearThread() {
  await fetch(`/thread/${threadId}/clear`, { method: 'POST' });
  log.innerHTML = '';
}

async function refreshUsage() {
  const r = await fetch('/usage');
  const j = await r.json();
  document.getElementById('usage').textContent = `req=${j.requests} tok=${j.total_tokens} ~$${j.cost_usd.toFixed(4)}`;
}
refreshUsage();

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const inp = document.getElementById('input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  inp.disabled = true;

  append(`<div class="turn"><div class="role-user">You: ${escapeHtml(text)}</div><div id="cur"></div></div>`);
  const cur = log.querySelector('.turn:last-child #cur');
  let assistantStarted = false;
  let thinkingNode = null;

  const resp = await fetch('/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ thread_id: threadId, message: text }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\\n\\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const evt = JSON.parse(line.slice(6));
      if (evt.type === 'reasoning_delta') {
        if (!thinkingNode) {
          cur.insertAdjacentHTML('beforeend', '<div class="thinking">Thinking: <span></span></div>');
          thinkingNode = cur.querySelector('.thinking:last-child span');
        }
        thinkingNode.textContent += evt.text;
      } else if (evt.type === 'content_delta') {
        if (!assistantStarted) {
          cur.insertAdjacentHTML('beforeend', '<div class="role-assistant">Assistant: <span></span></div>');
          assistantStarted = true;
        }
        const span = cur.querySelector('.role-assistant:last-child span');
        span.textContent += evt.text;
      } else if (evt.type === 'tool_start') {
        cur.insertAdjacentHTML('beforeend', `<div class="tool">→ ${evt.name}(${escapeHtml(JSON.stringify(evt.args))})</div>`);
        assistantStarted = false; thinkingNode = null;
      } else if (evt.type === 'tool_end') {
        const out = evt.output.length > 200 ? evt.output.slice(0, 200) + '...' : evt.output;
        cur.insertAdjacentHTML('beforeend', `<div class="tool-out">${escapeHtml(out)}</div>`);
      } else if (evt.type === 'error') {
        cur.insertAdjacentHTML('beforeend', `<div class="error">Error: ${escapeHtml(evt.message)}</div>`);
      }
      log.scrollTop = log.scrollHeight;
    }
  }
  inp.disabled = false; inp.focus();
  refreshUsage();
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """
    根路径返回单页 HTML / Serve single-page HTML at root.

    response_class=HTMLResponse 让 FastAPI 自动把返回串当 HTML 处理。
    response_class=HTMLResponse tells FastAPI to treat the return as HTML.
    """
    return _INDEX_HTML


# ─────────────────────────────────────────────────────────────────
# /chat: SSE 流式 / SSE streaming
# ─────────────────────────────────────────────────────────────────
class ChatBody(BaseModel):
    """
    Pydantic 请求体模型；FastAPI 自动校验 + 生成 OpenAPI doc。
    Pydantic body model; FastAPI auto-validates + generates OpenAPI.
    """
    thread_id: str
    message: str


@app.post("/chat")
async def chat(body: ChatBody) -> StreamingResponse:
    """
    触发一轮对话，SSE 流式返回事件。
    Trigger one turn; SSE stream the event sequence back.
    """
    async def gen():
        # 内层 async generator 把 agent 事件转 SSE 行。
        # Inner async generator: agent events → SSE lines.
        assert _store is not None
        async for event in run_turn(body.message, body.thread_id, _store):
            # SSE 协议格式：
            #   "data: <json>\n\n"
            # 每事件一行 data + 一个空行作为消息分隔符。
            # SSE format: "data: <json>\n\n" — one data line + blank separator.
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        # 终止标记：客户端看到 [DONE] 后停止读取。
        # Terminal marker; clients stop reading on [DONE].
        yield "data: [DONE]\n\n"

    # StreamingResponse 把异步生成器接到 HTTP body chunk 流。
    # StreamingResponse wires the async generator to the HTTP body stream.
    return StreamingResponse(gen(), media_type="text/event-stream")


# ─────────────────────────────────────────────────────────────────
# /thread/{tid}/clear & /usage
# ─────────────────────────────────────────────────────────────────
@app.post("/thread/{thread_id}/clear")
async def clear_thread(thread_id: str) -> dict:
    """
    清空某 thread_id 的历史 / Clear a thread's history.

    URL path 参数自动注入 thread_id 变量。
    URL path param auto-injected as the thread_id arg.
    """
    assert _store is not None
    await _store.clear(thread_id)
    return {"ok": True}


@app.get("/usage")
async def usage() -> JSONResponse:
    """
    当前累计 token 与成本（USAGE 全局单例）。
    Cumulative tokens + cost from the global USAGE singleton.
    """
    return JSONResponse(
        {
            "requests": USAGE.requests,
            "prompt_tokens": USAGE.prompt_tokens,
            "completion_tokens": USAGE.completion_tokens,
            "reasoning_tokens": USAGE.reasoning_tokens,
            "total_tokens": USAGE.total_tokens,
            "cost_usd": USAGE.cost_usd,
        }
    )


@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus 指标端点 / Prometheus scrape endpoint.

    返回 text/plain; version=0.0.4 格式数据。
    Returns text/plain; version=0.0.4 payload.

    generate_latest() 序列化所有已注册的 Counter / Histogram 为文本格式。
    generate_latest() serializes all registered metrics to the text format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────────────
# 启动入口 / Run entry
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    """
    `python -m src.web` 入口；生产建议用 uvicorn / gunicorn 部署。
    Entry for `python -m src.web`; prod should use uvicorn / gunicorn.
    """
    import uvicorn

    # reload=False：生产模式；开发可改 True 自动热重载。
    # reload=False for prod; True enables auto-reload during dev.
    uvicorn.run("src.web:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
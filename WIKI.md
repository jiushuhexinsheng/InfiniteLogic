# InfiniteLogic — Wiki

零框架 ReAct Agent 技术文档。**纯 httpx + Qdrant + FastEmbed**，无 LangChain / LangGraph / OpenAI SDK。

## 目录

1. [架构概览](#架构概览)
2. [核心概念](#核心概念)
3. [数据流](#数据流)
4. [模块详解](#模块详解)
5. [RAG 系统](#rag-系统)
6. [工具系统](#工具系统)
7. [记忆与会话管理](#记忆与会话管理)
8. [流式输出机制](#流式输出机制)
9. [DeepSeek 思考模式](#deepseek-思考模式)
10. [多 Agent 协作](#多-agent-协作)
11. [可观测性](#可观测性)
12. [Web UI 与 Streamlit](#web-ui-与-streamlit)
13. [安全设计](#安全设计)
14. [扩展指南](#扩展指南)
15. [常见问题](#常见问题)

---

## 架构概览

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────────────┐
│  CLI / FastAPI / Streamlit (三选一)                  │
│  - CLI: rich 渲染  |  Web: SSE 流  |  Streamlit     │
│  - 鉴权中间件 (API Key + 限流) → FastAPI             │
└──────────────┬───────────────────────────────────────┘
               │ user_input + thread_id
               ▼
┌──────────────────────────────────────────────────────┐
│  src.agent.run_turn (async generator)                │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │ LlmClient        │  │  TOOLS.acall(name)       │ │
│  │ · 重试+熔断      │─►│  · 工具并行化            │ │
│  │ · 连接池         │◄─│  · search_docs ────┐     │ │
│  │ · 缓存           │  │  · web_search      │     │ │
│  │ · Span 追踪       │  │  · calculator      │     │ │
│  └──────────────────┘  │  · sandbox (docker) │     │ │
│           │             └──────────────────────────┘ │
│  _trim_history + _summarize_history                  │
│           │                                          │
│  SessionStore (aiosqlite WAL + 分页 + TTL)           │
└──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  RAG 子系统 (search_docs 调用时)                     │
│  ┌────────────┐  ┌───────────┐  ┌────────┐         │
│  │ Qdrant     │+ │ BM25      │→ │ RRF    │         │
│  │ + FastEmb. │  │ + jieba   │  │ Fusion │         │
│  └────────────┘  └───────────┘  └────┬───┘         │
│                                       ▼             │
│                              CrossEncoder           │
│                              (BGE Reranker)          │
│  监控: watcher 自动入库 + health 健康检查            │
└──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  可观测性 / Observability                            │
│  · loguru → logs/*.log (含 Span JSON)                │
│  · prometheus-client → /metrics → Grafana            │
│  · tracing: Span → LLM call / tool execution         │
│  · 业务指标: TTFT / RAG hit rate / LLM refusals      │
└──────────────────────────────────────────────────────┘
```

ReAct 循环：**Reason → Act → Observe → Reason → ...** 由 `agent.run_turn` 的 `for step in range(recursion_limit)` 驱动，无图调度框架。

---

## 核心概念

### ReAct

LLM 在每轮决策：**继续调工具**（Act）或**直接回答**（终止）。`assistant_message.tool_calls` 非空时继续；空时退出循环。

### 消息格式

OpenAI 协议原生 dict，不引入框架抽象：

| role | 字段 | 用途 |
|------|------|------|
| `system` | `content` | SYSTEM_PROMPT |
| `user` | `content` | 用户输入 |
| `assistant` | `content`, `tool_calls?`, `reasoning_content?` | LLM 输出 |
| `tool` | `tool_call_id`, `content` | 工具结果 |

### thread_id

会话隔离键。`SessionStore` 按 `thread_id` 在 SQLite 里独立维护消息序列。

### tool_call ↔ tool_message 配对

assistant 发出 `tool_calls` 后必须紧跟同样数量的 `tool` role 消息（按 `tool_call_id` 对应）。`_trim_history` 用 `start_on="user"` + 丢弃孤儿 tool 消息保证配对完整。

### Streaming

`stream_chat` 是 `AsyncIterator[dict]`，每事件四种类型：
- `reasoning_delta` — 思考流（DeepSeek thinking 模式）
- `content_delta` — 最终回答流
- `tool_call_delta` — 工具调用片段（按 index 累积）
- `done` — 整轮结束，附完整组装的 assistant message

---

## 数据流

### 单轮无工具

```
HumanMessage("你好")
  → stream_chat → AssistantMessage("你好！...")
  → SessionStore.append
  → yield {type:"done"}
```

### 单工具调用

```
HumanMessage("当前北京时间")
  → stream_chat → Assistant(tool_calls=[get_current_datetime(tz_name="Asia/Shanghai")])
  → TOOLS.acall → "Current datetime ..."
  → 追加 ToolMessage 到 history
  → stream_chat 再走一轮 → Assistant("北京时间是 ...") → done
```

### RAG 检索流

```
HumanMessage("项目部署要求")
  → Assistant(tool_calls=[search_docs(query="部署要求")])
  → search_docs:
      Qdrant.search(k=20) ────────┐
                                  ├→ RRF fuse → CrossEncoder rerank(top_n=4)
      BM25.search(k=20)  ─────────┘
  → ToolMessage("[1] sample.txt\n...")
  → Assistant("根据文档，部署要求 ...") → done
```

### 多工具 + 多轮（并行）

```
HumanMessage("查一下文档里的 deadline，再算还剩几天")
  → Assistant(tool_calls=[search_docs(...)])
  → ToolMessage(...)
  → Assistant(tool_calls=[get_current_datetime(...), calculator(...)])
  ← 无副作用工具 asyncio.gather 并行执行
  → 多个 ToolMessage 顺序追加
  → Assistant("剩 7 天") → done
```

`recursion_limit=50` 防死循环。`AGENT_MAX_HISTORY_MESSAGES=80` + `_summarize_history` 防 context 爆炸。

---

## 模块详解

### `src/config.py`

`pydantic-settings.BaseSettings` 加载 `.env` → 模块级 `settings` 单例。字段名 snake_case ↔ env 大写自动映射。

关键配置：

| 字段 | 默认 | 说明 |
|------|------|------|
| `llm_api_key` | (必填) | OpenAI 协议 API key |
| `llm_base_url` | DeepSeek | 任何兼容端点 |
| `llm_model` | `deepseek-chat` | 模型 ID |
| `llm_thinking_enabled` | false | DeepSeek V4 思考模式 |
| `llm_retry_max` | 3 | 最大重试次数（0=不重试） |
| `llm_circuit_breaker_threshold` | 5 | 熔断连续失败阈值 |
| `llm_circuit_breaker_cooldown` | 30 | 熔断冷却秒数 |
| `agent_recursion_limit` | 50 | ReAct 最大循环 |
| `agent_max_history_messages` | 80 | trim_history 上限 |
| `agent_summarize_threshold` | 50 | 触发摘要压缩的消息数 |
| `agent_parallel_tools` | true | 无副作用工具并行执行 |
| `session_db_path` | `./sessions.db` | aiosqlite 文件 |
| `session_wal_enabled` | true | WAL 模式 |
| `session_ttl_days` | 30 | 会话过期天数 |
| `workspace_dir` | `./workspace` | 文件工具沙箱 |
| `sandbox_mode` | subprocess | Python 执行沙箱模式 |
| `rag_persist_dir` | `./qdrant_db` | Qdrant embedded 数据目录 |
| `rag_embedding_model` | `BAAI/bge-small-zh-v1.5` | FastEmbed ONNX |
| `rag_top_k_retrieval` / `rag_top_k_rerank` | 20 / 4 | 召回 / 精排 |
| `rag_hybrid_enabled` | true | BM25 + 向量混合 |
| `rag_reranker_enabled` | true | CrossEncoder 精排 |
| `rag_auto_ingest` | false | 文件监听自动入库 |
| `cache_enabled` | false | LLM 响应缓存开关 |
| `api_keys` | — | Web API Key（逗号分隔） |
| `auth_enabled` | true | Web 鉴权开关 |
| `tracing_enabled` | true | Span 追踪开关 |

### `src/llm.py` + `src/llm_client.py`

两层 LLM 调用架构：

**`llm.py`（底层）**：`stream_chat()` — 裸 httpx SSE 流式解析，支持传入共享 `AsyncClient` 实现连接池复用。

**`llm_client.py`（高层封装）**：
- **连接池** — 模块级 `httpx.AsyncClient` 单例（max_keepalive=10），消除重复 TCP/TLS 握手
- **分级重试** — 429 读 Retry-After、5xx/timeout 指数退避+jitter（最多 3 次）、4xx 不重试
- **熔断器** — CLOSED→OPEN（连续 5 次失败）→HALF_OPEN（冷却 30s）→CLOSED（3 次成功）
- **响应缓存** — TTL + LRU 精确匹配（SHA256 key），命中则跳过 LLM 调用

### `src/agent.py`

ReAct 循环主体。`async def run_turn(user_input, thread_id, session) -> AsyncIterator[dict]` 是核心。

生产化增强（P0-P2）：
- **重试+熔断** — 通过 `LlmClient.retry_stream_chat()` 替代直接调 `stream_chat()`
- **历史压缩** — `_summarize_history()` 超阈值时 LLM 摘要 + 删除旧消息
- **工具并行化** — 无副作用工具（search/read/calc/datetime）`asyncio.gather` 并行执行
- **批量提交** — 每 ReAct 步收集 assistant + tool msgs，一次 `append_many` 写库
- **TTFT 指标** — 首次 content_delta 时记录首 token 延迟

流程：
1. 从 `SessionStore` 拉历史 → 摘要压缩
2. 补 system + 新 user 消息，立即写库（crash 安全）
3. `for step in range(recursion_limit)`：
   - `_trim_history` 裁剪
   - `stream_chat` 拿 assistant message
   - 无 `tool_calls` → 终止
   - 有 → 并行执行（实际仍串行，可改 `asyncio.gather`），结果作 ToolMessage 追加
4. 全程 yield UI 事件 + Prometheus 计时埋点

### `src/session.py`

aiosqlite 会话存储。生产化增强（P1）：

- **WAL 模式** — `PRAGMA journal_mode=WAL`，并发读不阻塞写
- **分页加载** — `load_messages(offset, limit)` + `load_messages_paginated(page)`
- **批量提交** — `append_many()` 一次 commit 写入多条消息
- **会话 TTL** — `cleanup_expired()` 按 `session_ttl_days` 自动清理过期会话
- **按范围删除** — `delete_old_messages(thread_id, keep_recent)` 支持历史压缩

表 schema：

```sql
CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT    NOT NULL,
    idx        INTEGER NOT NULL,
    payload    TEXT    NOT NULL,    -- JSON-encoded message dict
    created_at REAL    NOT NULL
);
CREATE INDEX ix_messages_thread ON messages(thread_id, idx);
```

API：`SessionStore.open()` / `append` / `append_many` / `load_messages` / `load_messages_paginated` / `count_messages` / `delete_old_messages` / `cleanup_expired` / `clear` / `close`。

### `src/cli.py`

rich 终端 UI。`run_cli()` 主循环：
- 三态机控制 thinking / answer / tool 段落显示
- slash 命令分发：`/new` `/clear` `/reranker` `/hybrid` `/usage` `/team` `/help`
- 异常打印不杀进程

### `src/multi_agent.py`

Planner → Researcher → Writer 三阶段流水线：
- **Planner**：调 LLM 输出 JSON 子任务列表（2-5 项）
- **Researcher**：对每个子任务最多 3 轮工具循环，收集证据 → JSON
- **Writer**：综合 plan + findings 流式输出最终答案

各阶段独立 system prompt + 独立工具白名单。

### `src/usage.py`

`UsageStats` 累计 prompt / completion / reasoning token + 估算成本。`_PRICES` 是硬编码价目表（每千 token USD），按模型前缀匹配。

新 token 加入时同步写 Prometheus Counter（`TOKENS_TOTAL` / `COST_USD_TOTAL`）。

### `src/metrics.py`

prometheus-client 定义 7 个 Counter + 4 个 Histogram：

| 指标 | 类型 | labels |
|------|------|--------|
| `openbase_turns_total` | Counter | status |
| `openbase_turn_latency_seconds` | Histogram | — |
| `openbase_tool_calls_total` | Counter | name, status |
| `openbase_tool_latency_seconds` | Histogram | name |
| `openbase_tokens_total` | Counter | kind, model |
| `openbase_cost_usd_total` | Counter | model |
| `openbase_llm_errors_total` | Counter | kind |
| `openbase_ttft_seconds` | Histogram | — |
| `openbase_llm_refusals_total` | Counter | model |
| `openbase_rag_queries_total` | Counter | status |
| `openbase_rag_ingest_errors_total` | Counter | — |

`time_turn()` / `time_tool(name)` 是上下文管理器。

### `src/tracing.py`

轻量级 Span 追踪（零外部依赖）：

- `Span` 类 — trace_id / span_id / parent_id / duration_ms / attributes
- `trace_async_span()` — 异步上下文管理器（LLM 调用 / 工具执行）
- 使用 `contextvars` 存储当前 span（协程安全）
- 结构化 JSON 日志输出，可 grep/jq 分析或导入 Jaeger

### `src/auth.py`

Web 鉴权 + 限流：

- `AuthManager` — API Key SHA256 哈希验证 + 权限控制
- `RateLimiter` — 固定窗口限流（per-key, 60s）
- `AuthMiddleware` — ASGI 中间件，返回标准 401/403/429 + `X-RateLimit-*` 头

### `src/cache.py`

LLM 响应缓存：

- 精确匹配 — SHA256(model + JSON(messages) + JSON(tools)) 作为 key
- TTL 过期 + LRU 淘汰（达 `cache_max_entries` 上限时）
- 集成在 `LlmClient.retry_stream_chat()`：命中直接返回，成功写缓存

### `src/tools/sandbox.py`

Python 执行沙箱：

- 三模式：`subprocess`（默认）/ `docker` / `disabled`
- Docker 模式：`--network=none`、`--read-only`、`--cpus=0.5`、`--memory=128M`、`--pids-limit=50`
- Docker 不可用时自动降级到 subprocess

### `src/rag/watcher.py` + `src/rag/health.py`

RAG 监控：

- `RagWatcher` — 轮询 docs 目录 mtime 变更 → 自动触发 ingest（去重冷却 60s）
- `rag_health_check()` — collection 存在性、chunk 数、持久化目录、模型可用性
- `/rag/health` 端点 + Prometheus RAG 指标

### `src/logging_setup.py`

loguru 双 sink：
- `logs/openbase.log` 全量（按 10MB 轮转，保留 5 份）
- `logs/openbase.error.log` 仅 ERROR+
- 可选 stderr（默认关，避免污染 CLI）

---

## RAG 系统

### 整体流程

```
入库（ingest.py）：
  scan → SHA1 dedupe → load → split → embed → Qdrant upsert + BM25 add
                                                ↓
                                          manifest.json 记录

检索（search_docs）：
  query → Qdrant.search(top_k_retrieval)  ─┐
       → BM25.search(bm25_top_k)           ├→ RRF fuse
                                            ↓
                                  CrossEncoder rerank(top_k_rerank)
                                            ↓
                                  格式化输出（含编号/source/page）
```

### Qdrant embedded

- `QdrantClient(path="./qdrant_db")` 嵌入式模式，无需起服务
- Collection 名默认 `docs`，distance = COSINE
- 每个 chunk 的 point_id = `uuid5(NAMESPACE, "{sha1}:{chunk_idx}")` → 稳定可重入 upsert
- `payload` 含 `page_content` `metadata` `sha1` `chunk_idx`
- `sha1` 字段建 KEYWORD payload index → 按文件级删除快速

### BM25 (rank-bm25)

- jieba 分词支持中英混合
- 没有原生增删 API → 每次 add/remove 触发 `_rebuild()`
- `_docs` + `_shas` 并行数组追踪 chunk → sha1
- 持久化：`bm25.pkl`（BM25Okapi 对象）+ `bm25.json`（docstore）

### RRF (Reciprocal Rank Fusion)

```
score(d) = Σ 1 / (k + rank_i(d))
```

各路检索器结果按倒数排名相加。默认 `k=60`（Cormack 2009 推荐值）。无需归一化分数，跨检索器尺度差异不影响。

### CrossEncoder Reranker

`BAAI/bge-reranker-base`（~1.1GB）：
- 输入 `(query, passage)` 对列表
- `model.predict(pairs)` 返回分数数组
- 按分数降序取 `top_k_rerank`

替代选择：
| 模型 | 大小 | 备注 |
|------|------|------|
| `BAAI/bge-reranker-v2-m3` | ~2.3GB | 多语言更准 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~90MB | 英文极速 |

运行时切换：`/reranker model <hf_name>`。

### 原地更新

文件内容变化（SHA1 不同）：
1. `vs.delete_by_sha1(old_sha1)` 删 Qdrant 中所有旧 chunk
2. `bm25.remove_by_sha1(old_sha1)` 删 BM25 中所有旧 chunk
3. 用新 sha1 写入新 chunks
4. 更新 manifest

**不再需要 `--clear`** 全量重建。

### 文档加载

`src/rag/loader.py` 按扩展名分派：

| 格式 | Loader | 输出 |
|------|--------|------|
| `.txt` `.md` `.csv` | 直接 read_text | 一个 Document |
| `.pdf` | pypdf | 每页一个 Document（含 page metadata） |
| `.docx` `.doc` | docx2txt | 一个 Document |
| `.html` `.htm` | BeautifulSoup get_text | 一个 Document |

### 切分

`src/rag/splitter.py` 自实现递归切分（参考 LangChain RecursiveCharacterTextSplitter）：
- 按 `\n\n` → `\n` → ` ` → `""` 优先级
- chunk_size 上限 + chunk_overlap 重叠
- 80 行实现，无外部依赖

---

## 工具系统

### `@tool` 装饰器

```python
from src.tools.base import tool

@tool("Add two numbers")
def add(a: int, b: int) -> int:
    return a + b
```

自动从函数签名 + `typing.get_type_hints` 生成 OpenAI tool schema：

```json
{
  "type": "function",
  "function": {
    "name": "add",
    "description": "Add two numbers",
    "parameters": {
      "type": "object",
      "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"}
      },
      "required": ["a", "b"]
    }
  }
}
```

有默认值的参数自动从 `required` 排除。

### TOOLS 注册中心

`src.tools.base.TOOLS` 单例：

```python
TOOLS.schemas()           # list[dict]，喂给 LLM
TOOLS.call(name, args)    # 同步调用
await TOOLS.acall(name, args)  # 异步调用（同步函数走 to_thread）
```

异常统一兜底：`PermissionError` 透传 message，其他异常 `Error in <name>: ...`。

### 内置工具

| 工具 | 文件 | 说明 |
|------|------|------|
| `search_docs` | `rag_tool.py` | 混合 RAG 检索 |
| `web_search` `web_search_results` | `search.py` | DuckDuckGo |
| `calculator` | `calculator.py` | AST 白名单求值 |
| `get_current_datetime` | `datetime_tool.py` | IANA 时区 |
| `read_file` `write_file` `list_directory` | `file_tools.py` | workspace 沙箱 |
| `run_python_file` `exec_python_snippet` | `python_exec.py` | subprocess 隔离 |

### 添加新工具

1. 在 `src/tools/` 新建 `my_tool.py`：

```python
from src.tools.base import tool

@tool("Describe what this tool does (LLM sees this).")
def my_tool(param: str) -> str:
    return do_work(param)
```

2. 在 `src/tools/__init__.py` 加 import 触发注册：

```python
from src.tools import (
    ...,
    my_tool,  # 新增
)
```

3. （可选）在 `src/agent.py` SYSTEM_PROMPT 补一行说明。

---

## 记忆与会话管理

### 当前实现：aiosqlite SessionStore

每个 `thread_id` 维护独立消息序列。重启进程后用相同 thread_id 可继续对话。

```
sessions.db:
  thread_A: [system, user_1, assistant_1, tool_1, assistant_2, ...]
  thread_B: [system, user_1, assistant_1, ...]
```

### 历史裁剪

`agent._trim_history`：
- 始终保留首条 system message
- 截取最近 `AGENT_MAX_HISTORY_MESSAGES` 条
- 若 tail 起头是 tool message（孤儿） → 丢弃直到 user

### 升级到 PostgreSQL（生产）

替换 `src/session.py` 中的 aiosqlite 为 asyncpg / psycopg。表结构相同。

### 长会话摘要压缩（已实现 P1）

当历史消息超过 `agent_summarize_threshold`（默认 50 条）：
1. 取最旧 60% 消息拼成文本
2. 调 LLM 生成一段摘要（一次性调用，无工具）
3. 替换为 `[Previous conversation summary]` system 消息
4. 从 SQLite 删除旧消息

配置 `agent_summarize_keep_recent` 控制保留的最近消息数（默认 10）。失败自动回退不丢历史。

---

## 流式输出机制

### SSE (Server-Sent Events)

OpenAI 协议流式响应是 SSE：

```
data: {"choices":[{"delta":{"content":"He"}}]}\n\n
data: {"choices":[{"delta":{"content":"llo"}}]}\n\n
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"calc"}}]}}]}\n\n
data: [DONE]\n\n
```

`httpx.AsyncClient.stream + aiter_lines()` 解析。

### tool_calls 累积

流式中 tool_call 分多个 chunk：

```
chunk 1: {"index":0, "id":"call_x", "function":{"name":"calc"}}
chunk 2: {"index":0, "function":{"arguments":"{\"a\":"}}
chunk 3: {"index":0, "function":{"arguments":"1}"}}
```

`_accumulate_tool_calls` 按 `index` 分桶，字符串拼接。

### 事件层级

`stream_chat` yield 的事件：

| 事件 | 内容 |
|------|------|
| `reasoning_delta` | 思考流片段 |
| `content_delta` | 回答流片段 |
| `tool_call_delta` | 工具调用片段（UI 通常忽略） |
| `done` | 完整 assistant message |

CLI 与 Web UI 都消费同一事件流，逻辑统一。

---

## DeepSeek 思考模式

### 启用

`.env`：

```ini
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash
LLM_THINKING_ENABLED=true
LLM_REASONING_EFFORT=high
SHOW_REASONING=true
```

### 与 LangChain 版的差异

| 维度 | LangChain (ai/) | InfiniteLogic |
|------|-----------------|----------|
| reasoning_content 提取 | 需 monkey patch 三处 converter | 直接读 JSON 字段 |
| round-trip 回传 | 框架内部丢弃，需 patch 复原 | dict 原样塞回 messages list |
| 协议字段位置 | `extra_body.thinking` | body 顶层 `thinking` |

InfiniteLogic 的 `llm.py` 直接处理，**60 行 monkey patch 不复存在**。

### 行为

- `reasoning_content` 与 `content` 完全分离
- 工具调用轮次中 assistant 消息必须带回 `reasoning_content`，InfiniteLogic 通过 `additional_kwargs` 透传
- temperature 在 thinking 模式下被忽略，设了不报错

---

## 多 Agent 协作

### Supervisor 模式（硬编码三角色）

```
user_input
   ↓
Planner (LLM)
   ↓ JSON 子任务列表
Researcher (LLM + 工具循环)
   ↓ JSON findings
Writer (LLM 流式)
   ↓
final answer
```

### 触发

CLI：`/team <task>`
代码：`async for event in run_multi_agent(task): ...`

### 工具白名单

`_RESEARCHER_TOOLS = {search_docs, web_search, web_search_results, calculator, get_current_datetime, read_file, list_directory}`

Planner / Writer 不带工具。

### Researcher 工具循环

最多 3 轮：每轮 LLM 可发起 tool_calls → 收集 ToolMessage → 再让 LLM 决定继续或输出最终 findings JSON。

### 与单 Agent 对比

| 维度 | 单 Agent (`run_turn`) | 多 Agent (`run_multi_agent`) |
|------|---------------------|---------------------------|
| 适合 | 一般问答 / 编辑代码 | 调研 / 综述 / 多步推理 |
| LLM 调用数 | 1+ | 3+ |
| 会话持久化 | 写 SessionStore | 不写（一次性任务） |
| Token 成本 | 低 | 高（多次完整 system prompt） |

---

## 可观测性

### 日志（loguru）

- 文件：`logs/openbase.log`（全量）+ `logs/openbase.error.log`
- 轮转：10MB × 5 份
- 格式：`时间 | LEVEL | name:func:line | message`
- 配置：`LOG_LEVEL=INFO` / `LOG_TO_STDERR=false`

埋点位置：
- `agent.py` — tool_call 入参 + 结果长度 + LLM 错误栈
- 异常自动捕获 stacktrace（仅 ERROR sink）

### 指标（Prometheus）

`src/metrics.py` 定义：

| 指标 | 类型 | labels |
|------|------|--------|
| `openbase_turns_total` | Counter | status |
| `openbase_turn_latency_seconds` | Histogram | — |
| `openbase_tool_calls_total` | Counter | name, status |
| `openbase_tool_latency_seconds` | Histogram | name |
| `openbase_tokens_total` | Counter | kind, model |
| `openbase_cost_usd_total` | Counter | model |
| `openbase_llm_errors_total` | Counter | kind |

`time_turn()` / `time_tool(name)` 上下文管理器封装。

### `/metrics` 端点

FastAPI app 自动注册 `GET /metrics`，返回 `text/plain; version=0.0.4` 格式。Prometheus 抓取配置：

```yaml
scrape_configs:
  - job_name: openbase
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana 看板

`dashboards/openbase.json` 含 13 个面板分 4 区块（Turns / Tools / Tokens & Cost / Errors）。导入步骤见 `dashboards/README.md`。

---

## Web UI 与 Streamlit

### FastAPI Web UI

启动：`python web_server.py`（默认 `0.0.0.0:8000`）。

端点：
- `GET /` — 单页 HTML（vanilla JS，无前端框架）
- `POST /chat` — SSE 流式触发一轮
- `POST /thread/{tid}/clear` — 清空某会话
- `GET /usage` — JSON 统计
- `GET /metrics` — Prometheus

前端逻辑：
- `crypto.randomUUID()` 生成 thread_id
- `fetch + ReadableStream` 消费 SSE
- 实时渲染 thinking / content / tool 事件
- sidebar 显示 usage 实时刷新

### Streamlit UI

启动：`streamlit run streamlit_app.py`

特点：
- `st.chat_message` 渲染会话
- `st.session_state` 维护 thread_id 与日志
- `st.empty()` 占位符承载流式更新
- sidebar 含 hybrid / reranker toggle + usage 指标

适用：快速演示 / 内部工具。生产推荐 FastAPI 版（前后端解耦更灵活）。

---

## 安全设计

| 威胁 | 防护 |
|------|------|
| 任意代码执行（calculator） | AST 白名单，禁止函数调用 / 变量 / 属性 |
| 路径遍历（file tools） | `Path.relative_to` 严格判定（跨平台一致，避开 Windows 大小写陷阱） |
| Python 执行网络外连 | Docker 沙箱 `--network=none`（`SANDBOX_MODE=docker`） |
| Python 执行资源耗尽 | Docker 模式 CPU/内存/PID 限制；subprocess 模式超时兜底 |
| Python 执行提权 | `--security-opt=no-new-privileges` + 非 root 用户 |
| API Key 泄露 | `.env` 已在 `.gitignore`，永不进版本控制 |
| Web 端点滥用 | API Key SHA256 验证 + 固定窗口限流 + 标准速率头 |
| Agent 无限循环 | `AGENT_RECURSION_LIMIT=50` 兜底 + 熔断器自动切断 |
| LLM 调用失败 | 分级重试（指数退避+jitter）+ 熔断器保护 |
| 历史无限增长 | `_trim_history` 裁剪 + `_summarize_history` 摘要压缩 + 会话 TTL |
| 历史内存爆炸 | 分页加载 + 摘要压缩 + 旧消息 SQLite 物理删除 |
| 工具异常崩溃 | `TOOLS.acall` 统一捕获异常转字符串 |
| SQLite 写锁争用 | WAL 模式 + 批量提交（append_many） |
| BM25 pickle 反序列化 | 仅信任本地生成的 .pkl |
| Qdrant 文件锁 | `reset_client()` 显式关闭后再删持久化目录 |

## 测试

62 个单元测试覆盖核心模块：

| 模块 | 用例 | 覆盖重点 |
|------|------|---------|
| `test_calculator.py` | 17 | AST 安全求值 + 注入攻击防护 |
| `test_file_tools.py` | 10 | 路径穿越 + 文件 CRUD + 目录遍历 |
| `test_config.py` | 7 | 类型校验 + 边界值 + 新配置字段 |
| `test_agent.py` | 8 | 历史裁剪 + tool 配对保护 + 边界 |
| `test_llm.py` | 8 | Payload 构造 + SSE 解析 + Tool call 累积 |
| `test_session.py` | 8 | CRUD + 分页 + 批量写入 + TTL |

```bash
uv run pytest -v          # 运行全部 62 个测试
uv run pytest --cov=src   # 含覆盖率报告
```

---

## 扩展指南

### 切换 LLM 厂商

**Ollama 本地**：
```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
LLM_THINKING_ENABLED=false
```

**OpenAI**：
```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

**Anthropic（需经 OpenAI-protocol 代理，如 LiteLLM）**：
```ini
LLM_BASE_URL=http://localhost:4000  # litellm proxy
LLM_API_KEY=anthropic-key
LLM_MODEL=claude-sonnet-4-6
```

### 切换 embedding 模型

`.env`：

```ini
RAG_EMBEDDING_MODEL=BAAI/bge-m3   # 多语言 1024 维
```

**注意**：换模型后维度变化，必须 `python ingest.py --clear` 重建。

### 切换 Reranker

运行时：`/reranker model BAAI/bge-reranker-v2-m3`

或 `.env`：
```ini
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 切换到远端 Qdrant Cloud

修改 `src/rag/vectorstore.py:_get_client`：

```python
client = QdrantClient(
    url="https://your-cluster.qdrant.cloud",
    api_key=os.environ["QDRANT_API_KEY"],
)
```

### 添加 LangSmith / OpenTelemetry 追踪

未来集成路径：
- `httpx` 加 OTLP 中间件
- LLM 调用前后埋 trace span
- 兼容 OpenTelemetry Collector 上报

---

## 常见问题

**Q: 运行报 `LLM_API_KEY` 未设置**

`.env` 必须和 `main.py` 同目录，且含 `LLM_API_KEY=sk-xxx`。

**Q: ingest.py 首次很慢**

首次会下载：
- FastEmbed embedding ONNX（约 95MB）→ `~/.cache/fastembed`
- BGE Reranker（约 1.1GB）→ `~/.cache/huggingface`

之后秒启动。可在 `.env` 设 `RAG_RERANKER_ENABLED=false` 跳过 reranker 下载（精度略降）。

**Q: Agent 工具循环不停**

通常两类原因：
1. LLM 想做某事但缺对应工具（如想执行代码但只有 file_tools）→ 加工具或改 prompt
2. trim 太狠切断 tool 配对 → 调大 `AGENT_MAX_HISTORY_MESSAGES`

`AGENT_RECURSION_LIMIT=50` 是兜底，触发后看 `logs/openbase.log` 倒推循环工具。

**Q: 文档内容改了但检索结果还是旧的**

正常情况下原地更新已生效（按 sha1 删旧 + 写新）。若仍是旧结果：
- 检查 `qdrant_db/manifest.json` 中该文件 sha1 是否已更新
- 极端情况下 `python ingest.py --clear` 重建

**Q: DeepSeek 思考模式工具调用报 400**

不应该再发生（InfiniteLogic 不依赖框架，原生支持）。若发生：
- 确认 `LLM_BASE_URL` 是 DeepSeek 而非别的厂商
- 确认 `LLM_THINKING_ENABLED=true` 仅对 V4 模型开启

**Q: 切到 Ollama 后 RAG 还能用吗**

可以。RAG 不依赖 LLM，FastEmbed + Qdrant + Reranker 全本地。LLM 仅在最后综合答案时调用。

**Q: Web UI 多用户并发**

当前 `_store` 是单例 SessionStore，aiosqlite 支持并发读写但单连接性能有限。高并发场景换 PostgreSQL。

**Q: Streamlit 与 FastAPI 二选一**

- Streamlit：快速演示，单进程，无前后端拆分
- FastAPI：可对接任意前端 / 移动端 / 第三方系统，含 /metrics 可观测
推荐：原型用 Streamlit，部署用 FastAPI。

**Q: 多 Agent 模式适合什么场景**

适合需要"调研 + 综合"的任务，如：
- "调研 X 技术并写综述"
- "比较 A 和 B 的优缺点"

不适合简单问答（成本高 3x）。

**Q: 如何看每轮 token 消耗**

CLI：`/usage`
Web：sidebar 实时刷新
Prometheus：`openbase_tokens_total` 与 `openbase_cost_usd_total`

**Q: 怎么清空所有会话**

```bash
rm sessions.db  # 删整个 SQLite 文件
```

或 CLI 内 `/clear` 仅清当前 thread。

**Q: BM25 索引何时重建**

每次 `add_documents_with_sha1` / `remove_by_sha1` 自动重建（rank-bm25 无增量 API）。约 100K chunks 内可接受；规模再大要换 Tantivy / Lucene。

**Q: 在 CI 跑测试需要真 LLM API key 吗**

不需要。`tests/conftest.py` 注入 `LLM_API_KEY=test-key`，62 个单测覆盖纯函数（calculator / file_tools / config / session / agent / llm），不调 LLM。

---

## 相关文档

- [README.md](README.md) — 快速开始与使用指南
- [IMPROVEMENTS.md](IMPROVEMENTS.md) — 后续 25 项改进方向
- [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) — 生产化路线图（P0-P4 优先级矩阵）
- [dashboards/README.md](dashboards/README.md) — Grafana 看板导入说明

---

*Wiki 最后更新：2026-06-21（P0-P4 全线完工：重试熔断、Docker沙箱、鉴权限流、SQLite WAL+分页+TTL、历史压缩、RAG自动入库+监控、工具并行化、LLM缓存、分布式追踪、业务指标、62个测试用例）*
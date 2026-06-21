# InfiniteLogic — 生产化路线图 / Production Readiness Roadmap

> 基于对全部源代码的逐行审查，梳理将 InfiniteLogic 从「能跑的 ReAct Agent」推向「可信赖的生产服务」所需的工作。

---

## 目录

- [总览](#总览)
- [P0：可靠性与容错](#p0可靠性与容错)
  - [1. LLM 调用零重试机制](#1-llm-调用零重试机制)
  - [2. Python 执行沙箱仅隔离崩溃](#2-python-执行沙箱仅隔离崩溃)
  - [3. Web 端点无鉴权、无限流](#3-web-端点无鉴权限流)
- [P1：数据与状态的架构缺陷](#p1数据与状态的架构缺陷)
  - [4. SQLite 会话存储的单点瓶颈](#4-sqlite-会话存储的单点瓶颈)
  - [5. 会话历史无限增长](#5-会话历史无限增长)
  - [6. RAG 无增量更新、无监控](#6-rag-无增量更新无监控)
- [P2：性能与扩展性](#p2性能与扩展性)
  - [7. 工具调用串行执行](#7-工具调用串行执行)
  - [8. 每次 LLM 调用创建新 httpx 客户端](#8-每次-llm-调用创建新-httpx-客户端)
  - [9. 无 LLM 响应缓存](#9-无-llm-响应缓存)
- [P3：可观测性深化](#p3可观测性深化)
  - [10. 缺少分布式追踪](#10-缺少分布式追踪)
  - [11. 缺少关键业务指标](#11-缺少关键业务指标)
- [P4：测试与质量保障](#p4测试与质量保障)
  - [12. 几乎零测试覆盖](#12-几乎零测试覆盖)
- [优先级矩阵](#优先级矩阵)
- [建议的首批交付](#建议的首批交付)
- [与现有 IMPROVEMENTS.md 的关系](#与现有-improvementsmd-的关系)

---

## 总览

| 优先级 | 数量 | 类别 |
|--------|------|------|
| 🔴 P0  | 3    | 不改就崩 — 可靠性与容错 |
| 🟠 P1  | 3    | 数据与状态的架构缺陷 |
| 🟡 P2  | 3    | 性能与扩展性 |
| 🟢 P3  | 2    | 可观测性深化 |
| 🔵 P4  | 1    | 测试与质量保障 |

---

## 🔴 P0：可靠性与容错

### 1. LLM 调用零重试机制

**涉及文件：**
- `src/llm.py` — `stream_chat()` 函数
- `src/agent.py` — `run_turn()` 函数

**现状分析：**

`stream_chat()` 每次调用新建 `httpx.AsyncClient` 实例（第 190 行），通过 `client.stream("POST", ...)` 发起流式请求。一旦发生网络超时、4xx、5xx 或连接断开，异常直接传播到 `agent.py:222` 的 `except Exception` 分支：

```python
# src/agent.py:222 — 当前行为
except Exception as exc:
    logger.exception("LLM call failed at step {}", step)
    LLM_ERRORS_TOTAL.labels(kind=type(exc).__name__).inc()
    yield {"type": "error", "message": f"LLM call failed: {exc}"}
    TURNS_TOTAL.labels(status="error").inc()
    return  # ← 整轮对话终止，之前所做工具调用全部作废
```

**问题严重性：**
- 一个 flaky 网络抖动 → 整轮报废，用户必须从头输入
- 无区分临时错误（429 rate limit、502 网关超时）和永久错误（401 未授权、400 参数错）
- 无退避策略，重试即刻发起可能继续打爆限流
- 每次 stream_chat 都创建+销毁 httpx.AsyncClient，无连接池复用

**解决方案：**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Agent 层   │────▶│  LLM Client  │────▶│  Provider A  │
│  run_turn() │     │  (连接池)     │     │  (主)         │
└─────────────┘     │              │     └─────────────┘
                    │  重试策略     │       ↓ 失败
                    │  · 429: 等   │     ┌─────────────┐
                    │    Retry-    │────▶│  Provider B  │
                    │    After     │     │  (备)         │
                    │  · 5xx: 指数 │     └─────────────┘
                    │    退避+jitter│
                    │  · 4xx: 不重 │
                    │    试        │
                    │              │
                    │  熔断器      │
                    │  · 5次连续失 │
                    │    败 → 冷却 │
                    │  · 半开探测   │
                    └──────────────┘
```

**具体改动：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/llm_client.py` (新) | 封装连接池 + 重试 + 熔断 | ~200 行 |
| `src/config.py` | 加 `retry_max`、`retry_backoff`、`circuit_breaker_threshold`、`fallback_providers` | ~30 行 |
| `src/agent.py` | 区分类别：临时错误重试整步、永久错误终止 | ~40 行 |

**重试策略细节：**

| 错误类型 | 行为 |
|----------|------|
| 429 Too Many Requests | 读 `Retry-After` 头，等对应秒数后重试；无该头则指数退避 |
| 502 / 503 / 504 | 指数退避 + jitter，最多 3 次 |
| 连接超时 / 读取超时 | 指数退避 + jitter，最多 3 次 |
| 401 / 403 | 不重试，直接报错 |
| 400 / 422 | 不重试，直接报错 |

**熔断器状态机：**
```
CLOSED ──(连续5次失败)──▶ OPEN ──(冷却30s)──▶ HALF_OPEN
  ▲                                                │
  └────────(连续3次成功)────────────────────────────┘
                     │
                     └──(任意失败)──▶ OPEN
```

---

### 2. Python 执行沙箱仅隔离崩溃

**涉及文件：**
- `src/tools/python_exec.py` — `exec_python_snippet()` 和 `run_python_file()`

**现状分析：**

代码中已有自知之明的注释（第 16 行）：
```python
# ⚠️ subprocess 仅隔离崩溃；不限网络 / 不限导入；要更严需 Docker/Firejail/seccomp。
```

当前 `subprocess.run()` 只保证子进程崩溃不会挂掉主进程。LLM 生成的任意代码可以：

| 风险 | 说明 |
|------|------|
| 任意文件读取 | `open("/etc/passwd")` — 子进程继承主进程文件系统权限 |
| 网络外连 | `requests.post(exfil_server, data)` — 无网络隔离 |
| 资源耗尽 | `while True: threading.Thread(target=...).start()` — 只有 120s 硬超时 |
| 导入任意模块 | `import socket; import subprocess` — 无模块白名单 |
| 进程注入 | 可读写 `/proc`、`/dev` |

**解决方案：**

```
用户输入 → LLM 生成代码
              ↓
         ┌─────────────────────┐
         │   Docker Sandbox     │
         │  · no-network        │
         │  · read-only rootfs  │
         │  · tmpfs /tmp 64MB   │
         │  · cpus=0.5 mem=128M │
         │  · seccomp: default  │
         │  · timeout: 30s      │
         └─────────────────────┘
              ↓
         输出（stdout/stderr + exit code）
```

**具体改动：**

| 模块 | 改动 |
|------|------|
| `src/tools/sandbox.py` (新) | Docker 沙箱执行器：构建最小镜像、启动容器、抓输出、清理 |
| `src/config.py` | `SANDBOX_MODE` (docker / subprocess / disabled)、`SANDBOX_IMAGE`、`SANDBOX_MEMORY_LIMIT`、`SANDBOX_CPU_LIMIT` |
| `src/tools/python_exec.py` | 调用 sandbox 模块；保留 subprocess 作为 dev 模式 |

**Dockerfile.sandbox（最小执行镜像）：**
```dockerfile
FROM python:3.12-slim
RUN useradd -m -s /bin/false runner
USER runner
WORKDIR /sandbox
```

**docker run 参数：**
```
--network=none           # 无网络
--read-only              # 只读文件系统
--tmpfs /tmp:size=64M    # 临时空间
--cpus=0.5               # 半核
--memory=128M            # 128MB 上限
--pids-limit=50          # 进程数限制
--security-opt=no-new-privileges
--timeout 30             # 容器级超时
```

---

### 3. Web 端点无鉴权、无限流

**涉及文件：**
- `src/web.py` — FastAPI 应用

**现状分析：**

`src/web.py` 暴露了四个端点：
- `GET /` — HTML 页面（公开 OK）
- `POST /chat` — 触发 LLM 对话（**消耗 Money**）
- `POST /thread/{tid}/clear` — 清空会话
- `GET /usage` — 用量统计
- `GET /metrics` — Prometheus 指标

全部端点零鉴权、零限流。任何知道 URL 的人都能无限调用 `/chat` 烧 LLM 额度。

**解决方案：**

| 层 | 措施 |
|----|------|
| 认证 | Bearer Token / API Key 中间件 |
| 授权 | Per-key 权限：可设 `chat` / `admin` 等角色 |
| 限流 | Per-key / per-IP 速率限制 |
| 审计 | 每次 `/chat` 调用记录 key_id + IP + 时间戳 |

**具体改动：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/web.py` | 加入 `auth_middleware`、`rate_limit_middleware` | ~60 行 |
| `src/auth.py` (新) | API Key 验证 + 权限控制 | ~80 行 |
| `src/config.py` | `API_KEYS` (逗号分隔)、`RATE_LIMIT_PER_MINUTE` | ~15 行 |

**限流策略：**

```
每个 API Key:
  - 20 次 /chat / 分钟
  - 超过后返回 429 + Retry-After
  - 计数器用内存 dict（小规模）或 Redis（多副本）

无 Key 请求：
  - 返回 401 Unauthorized
  - 不计入限流（直接拒绝）
```

---

## 🟠 P1：数据与状态的架构缺陷

### 4. SQLite 会话存储的单点瓶颈

**涉及文件：**
- `src/session.py` — `SessionStore` 类

**现状分析：**

```python
# 每条消息立即 COMMIT — 写锁阻塞所有并发操作
async def append(self, thread_id: str, message: dict[str, Any]) -> None:
    ...
    await self._conn.commit()  # ← 写锁
```

ReAct 循环中每次 append 触发一次 commit —— 一轮对话可能产生 10-20 次 commit。多用户并发下，aiosqlite 的后台线程排队处理所有写入，读操作也被阻塞。

**问题清单：**

| 问题 | 现状 | 影响 |
|------|------|------|
| 写锁争用 | 每条消息一次 commit | 10 并发用户 → 严重排队 |
| 全量加载 | `load_messages` 无 LIMIT | 长会话 → 内存爆炸 |
| 无会话 TTL | 永不删除旧 thread | 数据库无限增长 |
| 无连接池 | 单连接 | aiosqlite 仅支持单后台线程 |

**解决方案：**

```
单机 / 小规模         ｜  多副本 / 规模化
──────────────────────┼─────────────────────
SQLite WAL 模式       ｜  PostgreSQL
+ 批量提交（每 turn）  ｜  + 连接池（asyncpg）
+ 消息分页 LAN         ｜  + 消息分页
+ 会话 TTL 清理        ｜  + 定时清理任务
```

**分阶段：**

| 阶段 | 改动 | 适用 |
|------|------|------|
| 阶段 1 | SQLite WAL 模式 + `append_many` 批量提交 | 当前即可做，零依赖 |
| 阶段 2 | 迁移到 PostgreSQL + asyncpg | 多副本部署前 |
| 阶段 3 | 消息分页 + 会话 TTL + 定时清理 | 与阶段 1/2 并行 |

---

### 5. 会话历史无限增长

**涉及文件：**
- `src/agent.py` — `_trim_history()` 函数
- `src/session.py` — `load_messages()` 函数

**现状分析：**

当前 `_trim_history()` 在发给 LLM 前裁剪上下文窗口，但 SQLite 里的历史**永不删除**。

一个 3 小时的支持对话：
- 用户 → AI → tool call → tool result → AI → ... （循环）
- 可能达 200+ 条消息
- `load_messages()` 无条件 `SELECT ... ORDER BY idx` 全量拉取
- 下次打开时 200 条消息全部加载到内存 + 全量发给 `_trim_history` 后再裁剪

**解决方案：**

```
短对话（<50条）          ｜  长对话（>80条）
─────────────────────────┼─────────────────────
保持现状                  ｜  触发摘要压缩
                          ｜  1. 取出最旧 40%
                          ｜  2. 调 LLM 生成一段摘要
                          ｜  3. 替换为 system 消息:
                          ｜     "Previous conversation
                          ｜      summary: {摘要}"
                          ｜  4. 从 SQLite 删除旧消息
```

**具体改动：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/agent.py` | `_summarize_history()` 函数：超过阈值时调 LLM 摘要 | ~60 行 |
| `src/session.py` | `delete_old_messages()` 方法：按 idx 范围删除 | ~30 行 |
| `src/config.py` | `AGENT_SUMMARIZE_THRESHOLD`、`AGENT_SUMMARIZE_KEEP_RECENT` | ~10 行 |

---

### 6. RAG 无增量更新、无监控

**涉及文件：**
- `src/rag/ingest_pipeline.py` — `ingest()` 函数
- `src/rag/vectorstore.py` — Qdrant 操作

**现状分析：**

当前 RAG 知识库完全靠**手动运行** `python ingest.py` 来更新。生产场景下文档持续变化，需要更自动化的方式。

**缺口：**

| 缺失能力 | 说明 |
|----------|------|
| 文件系统 Watch | 文档变更后自动触发摄入 |
| 检索质量监控 | 无回报命中率、空结果率 |
| 向量库健康检查 | collection 是否存在、chunk 数是否异常归零 |
| 摄入失败告警 | 文档解析失败静默跳过，无告警 |

**解决方案：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/rag/watcher.py` (新) | watchdog 监听 docs 目录 → 自动调用 ingest | ~80 行 |
| `src/rag/health.py` (新) | 向量库健康检查 API：chunk 数、collection 状态 | ~50 行 |
| `src/metrics.py` | 加 `RAG_QUERIES_TOTAL`、`RAG_EMPTY_RESULTS`、`RAG_INGEST_ERRORS` | ~30 行 |
| `src/config.py` | `RAG_AUTO_INGEST` | ~5 行 |

---

## 🟡 P2：性能与扩展性

### 7. 工具调用串行执行

**涉及文件：**
- `src/agent.py:263` — ReAct 主循环中的 tool_call 执行

**现状分析：**

代码中已标记：
```python
# 当前实现：串行。可改 asyncio.gather 并行（注意 file_write 等有副作用工具需互斥）。
```

LLM 一次返回 3 个彼此独立的 tool_call（如查文档 + 搜索网络 + 计算），目前必须排队等。总延迟 = T1 + T2 + T3 而非 max(T1, T2, T3)。

**解决方案：**

```python
# 并行执行，但标记有副作用的工具需互斥
SAFE_PARALLEL = {"search_docs", "web_search", "web_search_results",
                 "calculator", "get_current_datetime", "read_file", "list_directory"}
NEEDS_MUTEX = {"write_file", "run_python_file", "exec_python_snippet"}

# 并行跑无副作用工具，串行跑有副作用工具
parallel_results = await asyncio.gather(*[TOOLS.acall(tc) for tc in safe_calls])
```

需注意：DeepSeek/OpenAI 协议中 tool_call 有顺序语义（后面的可能依赖前面的结果），并行化需要语义感知 —— 完全独立 → 并行，有依赖 → 保持顺序。

**具体改动：** `src/agent.py` ~60 行。

---

### 8. 每次 LLM 调用创建新 httpx 客户端

**涉及文件：**
- `src/llm.py:190` — `async with httpx.AsyncClient(...)`

**现状分析：**

每次 `stream_chat()` 调用都创建并销毁一个 `AsyncClient`。一次 ReAct 循环平均 3-5 次 LLM 调用 = 3-5 次 TCP 握手 + TLS 协商。在网络条件不佳时，TLS 握手可能比实际响应时间还长。

**解决方案：**

```python
# src/llm.py — 改为模块级连接池
_CLIENT: httpx.AsyncClient | None = None

async def get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_request_timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _CLIENT
```

注意：httpx.AsyncClient 的 `stream()` 返回的上下文管理器退出时**不会**关闭底层连接 —— 它依靠 `Keep-Alive` 复用。

**具体改动：** `src/llm.py` ~30 行。

---

### 9. 无 LLM 响应缓存

**现状分析：**

相同 query 进两次 → 两次完整 ReAct 循环 → 两次 LLM 费用。这在以下场景尤为浪费：
- 用户反复问相同或类似问题
- 开发调试期间反复跑同一请求
- 多用户问相同领域问题

**解决方案：**

| 层级 | 策略 | 适用 |
|------|------|------|
| 精确缓存 | query 字符串哈希 → 检查是否完全匹配 | 完全相同的请求 |
| 语义缓存 | Embedding 相似度 > 0.95 → 返回缓存 | 相似但不完全相同的请求 |
| 步骤缓存 | 相同的 tool call → tool result 缓存（短 TTL） | ReAct 循环内 |

**具体改动：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/cache.py` (新) | 精确缓存 + 语义缓存（可选） | ~100 行 |
| `src/agent.py` | 在 `run_turn()` 入口处查缓存 | ~20 行 |
| `src/config.py` | `CACHE_ENABLED`、`CACHE_TTL` | ~10 行 |

---

## 🟢 P3：可观测性深化

### 10. 缺少分布式追踪

**涉及文件：**
- `src/metrics.py` — Prometheus 指标

**现状分析：**

当前 Prometheus 指标是聚合级别的 —— 你能看到 "P95 工具调用耗时 10s"，但无法回答：
- 具体哪个会话的哪一步慢？
- 慢在哪一层的哪个操作？
- 多个服务调用之间的因果关系？

**解决方案：**

采用 OpenTelemetry 标准，在关键路径上打 Span：

```
Turn (trace_id=xxx)
  ├── Step 1
  │   ├── LLM call (span)
  │   └── Tool: search_docs (span)
  │       ├── Qdrant query (span)
  │       ├── BM25 query (span)
  │       └── Rerank (span)
  ├── Step 2
  │   ├── LLM call (span)
  │   └── Tool: calculator (span)
  └── Step 3
      └── LLM final (span)
```

**具体改动：**

| 模块 | 改动 | 行数估计 |
|------|------|----------|
| `src/tracing.py` (新) | OpenTelemetry 初始化 + Span 辅助函数 | ~60 行 |
| `src/agent.py` | 每个 step / tool 包 Span | ~30 行 |
| `src/llm.py` | LLM 调用 Span | ~20 行 |
| `pyproject.toml` | 加 `opentelemetry-api`、`opentelemetry-exporter-otlp` | - |

**导出后端：** Jaeger / Grafana Tempo / Datadog（按部署环境选）

---

### 11. 缺少关键业务指标

**现状分析：**

当前指标偏基础设施（计数 + histogram），缺少业务视角：

| 缺失指标 | 价值 |
|----------|------|
| 每对话成本 | 知道具体会话花了多少钱 |
| 用户活跃度 | 日活 / 周活 |
| 工具使用分布 | 知道哪个工具最常用 |
| TTFT (Time To First Token) | 用户感知的"响应速度" |
| 首轮解决率 | 不用回退/重试就成功的比例 |
| RAG 命中率 | 多少 query 在知识库找到了结果 |
| LLM 拒绝率 | LLM 返回内容过滤 / 拒答次数 |

**具体改动：** `src/metrics.py` ~50 行；Web UI `/usage` 端点的展示增强。

---

## 🔵 P4：测试与质量保障

### 12. 几乎零测试覆盖

**现状分析：**

`tests/` 目录只有两个文件：
- `__init__.py` — 空
- `conftest.py` — 注入 `LLM_API_KEY=test-key`

**需要的测试层次：**

```
            ╱─────────────╲
           ╱  E2E 测试     ╲   少量、慢、真实 LLM
          ╱─────────────────╲
         ╱   集成测试          ╲  中等、Mock LLM
        ╱───────────────────────╲
       ╱   单元测试                ╲ 大量、快、Mock everything
      ╱─────────────────────────────╲
     ╱   静态分析 (ruff + mypy)       ╲  已有，保持
    ╲─────────────────────────────────╱
```

**单元测试覆盖清单：**

| 模块 | 测试内容 | 优先级 |
|------|----------|--------|
| `src/tools/calculator.py` | 合法/非法表达式、边界值 | 高 |
| `src/tools/file_tools.py` | 读/写/列表 + 路径穿越攻击 | 高 |
| `src/tools/python_exec.py` | 正常执行、超时、语法错误 | 高 |
| `src/tools/search.py` | 搜索结果解析 | 中 |
| `src/session.py` | CRUD 操作、并发追加 | 高 |
| `src/llm.py` | SSE 解析、payload 构造、tool_call 累积 | 高 |
| `src/agent.py` | `_trim_history` 边界情况 | 高 |
| `src/config.py` | 配置校验 | 中 |
| `src/usage.py` | Token 累加、成本计算、未知模型降级 | 中 |
| `src/tools/base.py` | Schema 生成、sync/async 派发 | 中 |
| `src/rag/` | 去重逻辑、分割、RRF 融合 | 中 |

**集成测试清单：**

| 场景 | 做法 |
|------|------|
| 完整 ReAct 循环 | Mock `stream_chat` 返回预设回复，验证 tool 执行和事件流 |
| 工具调用链 | Mock LLM 返回带 tool_calls 的 message，验证多步循环 |
| 错误恢复 | Mock LLM 抛异常 → 验证优雅降级 |
| 历史裁剪 | 构造超长 history → 验证裁剪后无孤儿 tool 消息 |

**安全测试清单：**

| 攻击向量 | 测试方法 |
|----------|----------|
| 路径穿越 | `../../etc/passwd`、`C:\Windows\...`、符号链接 |
| Python 沙箱逃逸 | `__import__('os')`、`eval`、`exec`、`compile` |
| 计算器注入 | `__import__`、属性访问、大数 DOS |
| JSON 注入 | 畸形 tool_call arguments |
| Prompt 注入 | User 消息伪装 system prompt |
| SSE 注入 | LLM 返回内容含 `data:` 行 |

---

## 优先级矩阵

```
                    影响面
            高              低
        ┌──────────────┬──────────────┐
实  高  │ 1.LLM重试熔断 │ 4.会话TTL    │
施      │ 2.Python沙箱  │ 6.RAG自动更新│
难      │ 3.鉴权限流    │              │
度  低  ├──────────────┼──────────────┤
        │ 5.历史压缩    │ 7.工具并行化  │
        │ 8.HTTP连接池  │ 9.LLM缓存    │
        │ 10.分布式追踪 │ 11.业务指标  │
        │ 12.测试覆盖   │              │
        └──────────────┴──────────────┘
```

---

## 建议的首批交付

**冲刺 1（1-2 周）：安全底线**

| 事项 | 理由 |
|------|------|
| Web 鉴权 + 限流 | 不对公网暴露裸端点，是第一条防线 |
| Python 沙箱 Docker 化 | 代码注释自己都在警告不安全 |
| 配置交叉校验 | 一行 pydantic validator 防低级配错 |

**冲刺 2（1-2 周）：可靠性**

| 事项 | 理由 |
|------|------|
| LLM 客户端重构（连接池 + 重试 + 熔断） | 影响所有请求，不改就随时崩 |
| SQLite WAL + 批量提交 | 零依赖改进，立即生效 |
| 会话 TTL + 分页加载 | 防止磁盘/内存泄漏 |

**冲刺 3（1-2 周）：质量保障**

| 事项 | 理由 |
|------|------|
| 工具层单元测试 | 最核心的 6 个工具先覆盖 |
| Mock LLM 集成测试 | 验证 ReAct 循环正确性 |
| 关键业务指标 | 在 Grafana 能看到具体会话的成本和延迟 |

**冲刺 4（按需）：体验优化**

| 事项 | 理由 |
|------|------|
| 工具并行化 | 延迟明显下降 |
| 分布式追踪 | 定位性能瓶颈 |
| 历史摘要压缩 | 长对话性能 |
| LLM 缓存 | 成本控制 |

---

## 与现有 IMPROVEMENTS.md 的关系

IMPROVEMENTS.md (25 项) 与本路线图互补：

| IMPROVEMENTS.md 已覆盖 | 本文新增 / 补强 |
|------------------------|----------------|
| #2 Docker Compose（Dockerfile 缺） | P0-1 LLM 重试熔断（**关键缺失**） |
| #6 Docker sandbox（草案） | P0-2 sandbox 安全细节（**补强**） |
| #7 Qdrant Cloud | P0-3 鉴权限流（**关键缺失**） |
| #12 工具并行化 | P1-4 SQLite → PG（**补强**） |
| #15 CLI 命令扩展 | P1-5 历史压缩（**补强**） |
| #21 长会话摘要 | P1-6 RAG 自动更新（**补强**） |
| | P2-8 HTTP 连接池（**新**） |
| | P2-9 LLM 缓存（**新**） |
| | P3-10 分布式追踪（**新**） |
| | P3-11 业务指标（**新**） |
| | P4-12 测试体系（**关键缺失**） |

**IMPROVEMENTS.md 缺少的最关键三项：LLM 重试熔断、鉴权限流、测试体系。**

---

*文档生成于 2026-06-14，基于对全部源代码的审查。*

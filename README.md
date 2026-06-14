# InfiniteLogic

零框架 ReAct Agent。**纯 httpx + Qdrant + FastEmbed**。无 LangChain、无 LangGraph、无 OpenAI SDK。

兼容任何 OpenAI 协议接口（DeepSeek / Ollama / LiteLLM / 自建 vLLM）。

## 设计哲学

- **依赖最少** — 19 个核心包，无框架层抽象
- **完全可控** — 每个字节怎么走、每条消息怎么存全由本仓决定
- **DeepSeek thinking 原生** — 不再需要 monkey patch，`reasoning_content` 直通
- **全本地 RAG** — FastEmbed ONNX + Qdrant + BGE Reranker，零外部 API

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
uv sync

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. （可选）入库文档
python ingest.py

# 4. 启动 Agent
python main.py
```

> 国内用户如遇网络超时，设置镜像源：
> ```powershell
> $env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
> uv sync
> ```

## 功能特性

- **ReAct Agent 循环** — 纯 while 循环驱动，无图调度框架
- **流式输出** — SSE 解析，token 级实时渲染（含 DeepSeek 思考流）
- **RAG 知识库** — FastEmbed + Qdrant + BGE Reranker 两段检索
- **SHA1 去重入库** — 防止重复 ingest 膨胀索引
- **会话持久化** — aiosqlite 写 `sessions.db`，重启不丢
- **历史自动裁剪** — 保留 system + 最近 N 条，不切断 tool_call 配对
- **10 个内置工具** — RAG / 网搜 / 计算 / 时间 / 文件读写 / Python 执行
- **安全沙箱** — 计算器 AST 白名单；文件工具 `Path.relative_to` 严格判定；Python 执行 subprocess + 超时

## CLI 命令

| 输入 | 效果 |
|------|------|
| 任意文字 | 发送给 Agent |
| `/new` | 开启新会话（换 thread_id） |
| `/clear` | 清空当前会话历史 |
| `/reranker on\|off` | 开关精排 |
| `/reranker model <hf_name>` | 热切换精排模型 |
| `/hybrid on\|off` | 开关 BM25 + 向量混合检索 |
| `/usage` | 显示累计 token / 成本 |
| `/usage reset` | 重置计数 |
| `/team <task>` | 跑多 Agent 流水线 |
| `/help` | 命令帮助 |
| `/quit` `/exit` | 退出 |
| `Ctrl+C` | 强制退出 |

## Web UI

```bash
python web_server.py            # 0.0.0.0:8000
HOST=127.0.0.1 PORT=8080 python web_server.py
```
打开 `http://localhost:8000` 即用。功能：流式聊天、思考流显示、工具调用可视化、token/cost 实时刷新、新会话与清空。

## 多 Agent 流水线

```
You> /team 调研 DeepSeek V4 的思考模式并写一段中文综述
```

阶段 / Stages:
1. **Planner** — LLM 把任务拆成 2-5 个 JSON 子任务
2. **Researcher** — 对每个子任务调 `search_docs` / `web_search` / `calculator` 等工具收集证据（最多 3 轮工具循环）
3. **Writer** — 综合 plan + findings 流式输出最终答案

各阶段独立 system prompt + 独立工具白名单。

## 测试与 CI

```bash
# 安装开发依赖（含 mypy / pytest / ruff）
uv sync --extra dev

# Lint
uv run ruff check src tests

# 静态类型检查
uv run mypy

# 单测
uv run pytest -v
```

CI（`.github/workflows/ci.yml`）：push / PR 触发，矩阵 Python 3.10 / 3.11 / 3.12，跑 ruff + mypy + pytest。

### 类型检查策略

- **PEP 604 注解**：`dict[str, Any]` / `list[X] | None` 全用上
- **mypy 起步宽**：`strict = false`，渐进收紧
- **第三方 stubs**：通过 `ignore_missing_imports` 跳过 fastembed / qdrant / jieba 等无 stubs 库
- **逐模块严格化**：`pyproject.toml` 里的 `[[tool.mypy.overrides]]` 可单独放宽 / 收紧某些包

## 内置工具

| 工具 | 功能 |
|------|------|
| `search_docs` | 检索本地知识库（优先调用） |
| `web_search` | DuckDuckGo 搜索，返回合并文本 |
| `web_search_results` | DuckDuckGo 搜索，返回 JSON 结构（含 URL） |
| `calculator` | AST 白名单表达式求值 `+ - * / ** % //` |
| `get_current_datetime` | 任意 IANA 时区当前时间 |
| `read_file` | 读 workspace 内文件 |
| `write_file` | 写 workspace 内文件 |
| `list_directory` | 列 workspace 目录内容 |
| `run_python_file` | 执行 workspace 内 .py 文件 |
| `exec_python_snippet` | 执行临时 Python 代码片段 |

## 配置说明（`.env`）

```ini
# --- LLM ---
LLM_API_KEY=sk-xxx                              # 必填
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
LLM_REQUEST_TIMEOUT=120

# DeepSeek thinking 模式（仅 V4 模型）
LLM_THINKING_ENABLED=false
LLM_REASONING_EFFORT=high
SHOW_REASONING=true

# --- Agent ---
AGENT_RECURSION_LIMIT=50
AGENT_MAX_HISTORY_MESSAGES=80

# --- 会话 ---
SESSION_DB_PATH=./sessions.db

# --- 沙箱 ---
WORKSPACE_DIR=./workspace

# --- RAG ---
RAG_DOCS_DIR=./docs
RAG_PERSIST_DIR=./qdrant_db
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=64
RAG_TOP_K_RETRIEVAL=20
RAG_TOP_K_RERANK=4
RAG_RERANKER_ENABLED=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
```

### 切换厂商

**Ollama（本地）**
```ini
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
LLM_THINKING_ENABLED=false
```

**OpenAI**
```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
```

**DeepSeek V4 思考模式**
```ini
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash
LLM_THINKING_ENABLED=true
LLM_REASONING_EFFORT=high
```

## RAG 知识库

```bash
python ingest.py                    # 入库 docs/ 全部支持文件
python ingest.py path/to/file.pdf   # 单文件
python ingest.py --clear            # 删除整个索引重建
```

支持格式：`.txt` `.md` `.pdf` `.docx` `.csv` `.html`

去重：`qdrant_db/manifest.json` 记录每文件 SHA1，未变化跳过。内容变化自动原地更新（按 SHA1 删旧 chunk + 写新 chunk）。

检索流程：
1. FastEmbed 编码 query
2. **召回**（当前串行，可改为并行）：
   - Qdrant 余弦相似度 → `RAG_TOP_K_RETRIEVAL` 条
   - BM25（jieba 分词）→ `RAG_BM25_TOP_K` 条
3. RRF 融合两路排名
4. BGE CrossEncoder Reranker 精排 → `RAG_TOP_K_RERANK` 条

首次运行会自动下载：
- FastEmbed embedding ONNX（约 95MB） → `~/.cache/fastembed`
- BGE Reranker（约 1.1GB） → `~/.cache/huggingface`

**原地更新**：再次 ingest 同一文件且内容变化时，自动按旧 sha1 删除全部旧 chunk（Qdrant + BM25 同步），再以新 sha1 写入。**不再需要 `--clear` 全量重建**。

## 项目结构

```
InfiniteLogic/
├── main.py               # 程序入口（async bootstrap）
├── ingest.py             # 文档入库 CLI
├── pyproject.toml        # 项目元数据 + 依赖声明 + 工具配置
├── uv.lock               # 锁定所有依赖精确版本（不可手动编辑）
├── requirements.txt      # 已废弃 → 迁移至 pyproject.toml
├── requirements-dev.txt  # 已废弃 → 迁移至 pyproject.toml
├── README.md             # 快速开始与使用指南
├── WIKI.md               # 技术文档（架构/模块/扩展指南）
├── IMPROVEMENTS.md       # 25 项后续改进方向
├── .env.example          # 配置模板
├── .gitignore
├── docs/                 # 待入库文档目录
├── qdrant_db/            # Qdrant 索引 + manifest.json（自动创建）
├── sessions.db           # aiosqlite 会话持久化（自动创建）
├── workspace/            # 文件工具沙箱（自动创建）
└── src/
    ├── config.py         # 配置（pydantic-settings）
    ├── llm.py            # httpx LLM 客户端（流式 SSE 解析 + usage 累计）
    ├── session.py        # aiosqlite 会话存储
    ├── agent.py          # ReAct 循环 + 历史裁剪
    ├── cli.py            # rich 终端 UI + slash 命令
    ├── multi_agent.py    # Planner / Researcher / Writer 流水线
    ├── web.py            # FastAPI + SSE 单页前端
    ├── usage.py          # token & cost 累计
    ├── logging_setup.py  # loguru 配置
    ├── metrics.py        # Prometheus 计数器 / 直方图
    ├── rag/
    │   ├── document.py        # 最小 Document 模型
    │   ├── splitter.py        # 递归文本切分
    │   ├── loader.py          # 文档加载（pdf/docx/...）
    │   ├── vectorstore.py     # FastEmbed + Qdrant（embedded）原地删除
    │   ├── bm25_index.py      # BM25 索引（jieba 分词，支持 sha1 删除）
    │   ├── fusion.py          # RRF 多路融合
    │   └── ingest_pipeline.py # SHA1 去重 + 原地更新
    └── tools/
        ├── base.py            # @tool 装饰器 + 注册中心
        ├── calculator.py
        ├── datetime_tool.py
        ├── file_tools.py
        ├── python_exec.py     # run_python_file / exec_python_snippet
        ├── rag_tool.py        # 混合检索 + 热切换接口
        └── search.py

tests/                    # pytest 单测
.github/workflows/ci.yml  # GitHub Actions
web_server.py             # FastAPI Web UI 入口
streamlit_app.py          # Streamlit UI 入口
```

## 依赖一览

| 包 | 用途 |
|----|------|
| `httpx` | LLM HTTP 调用 + SSE 流 |
| `pydantic` / `pydantic-settings` | 数据校验 + 配置加载 |
| `aiosqlite` | 异步 SQLite 会话存储 |
| `rich` | 终端渲染 |
| `fastembed` | 本地 ONNX embedding |
| `qdrant-client` | 向量索引（embedded 模式，支持 metadata 删除） |
| `rank-bm25` / `jieba` | 关键词检索 + 中文分词 |
| `sentence-transformers` | BGE Reranker (CrossEncoder) |
| `loguru` | 结构化日志 |
| `fastapi` / `uvicorn` | Web UI + SSE |
| `streamlit` | 备选 UI |
| `prometheus-client` | /metrics 埋点 |
| `pypdf` / `docx2txt` / `beautifulsoup4` | 文档解析 |
| `duckduckgo-search` | 网页搜索 |

**没有：** LangChain、LangGraph、OpenAI SDK、LlamaIndex。

## 安全设计

| 威胁 | 防护 |
|------|------|
| 任意代码执行（calculator） | AST 白名单，禁止函数调用 / 变量 / 属性 |
| 路径遍历（file tools） | `Path.relative_to` 严格判定，跨平台一致 |
| Python 执行越界 | subprocess + workspace cwd + 30s/120s 超时 + 输出截断 |
| API Key 泄露 | `.env` 已在 `.gitignore`，不会进版本控制 |
| Agent 无限循环 | `AGENT_RECURSION_LIMIT=50` 兜底 |
| 历史无限增长 | `_trim_history` 保留 system + 最近 N 条 |
| 工具异常崩溃 | `TOOLS.acall` 统一捕获异常转字符串 |
| Qdrant 文件锁 | `reset_client()` 显式关闭后再删持久化目录 |
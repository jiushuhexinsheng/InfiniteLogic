# InfiniteLogic — 改进方向 / Improvement Roadmap

> 已完成 13 次迭代（v0.13），本文档列出下一阶段可做的事项。
> 13 iterations shipped (v0.13). This doc lists what could come next.

---

## 已完成 / Shipped (v0.1 → v0.13)

- [x] v0.1  — 基础 ReAct + 工具系统 + RAG + CLI
- [x] v0.2  — 双语注释 + 安全沙箱（is_relative_to / AST 白名单）
- [x] v0.3  — AsyncSqliteSaver 会话持久化 + trim_messages
- [x] v0.4  — Python 执行工具（file + snippet）
- [x] v0.5  — OpenBase 重写（无 LangChain，纯 httpx）
- [x] v0.6  — BM25 混合检索 + RRF 融合
- [x] v0.7  — Reranker 热切换 + /hybrid /reranker 命令
- [x] v0.8  — Token & 成本统计 + /usage 命令
- [x] v0.9  — loguru 结构化日志
- [x] v0.10 — FastAPI Web UI + SSE
- [x] v0.11 — 多 Agent 协作（Planner / Researcher / Writer）
- [x] v0.12 — pytest + GitHub Actions CI 矩阵
- [x] v0.13 — Qdrant 原地删除 + BM25 SHA1 删除 + Reranker 独立缓存 + Streamlit UI + Prometheus /metrics + Grafana dashboard

---

## 🔴 高价值 / 低成本

### 1. Grafana Provisioning + Alerting
- **目标**：开箱即用的告警规则
- **做法**：
  - `dashboards/provisioning/dashboards.yml` + `datasources.yml`
  - `alerts.yml` 定义：错误率 > 20% / 单轮 p95 > 60s / 单小时成本 > $1
- **依赖**：Grafana 10+
- **估算**：~150 行 YAML

### 2. Docker Compose 一键部署
- **目标**：`docker compose up` 即可跑全栈（InfiniteLogic + Prometheus + Grafana）
- **做法**：
  - `Dockerfile`（Python 3.12-slim + uv 安装依赖）
  - `docker-compose.yml`（3 service：openbase / prometheus / grafana，含 provisioning volume）
  - `.dockerignore`
- **估算**：~80 行配置

### 3. 多 Agent 流水线指标埋点
- **目标**：观察 Planner / Researcher / Writer 各阶段耗时与失败率
- **做法**：
  - `metrics.py` 加 `MULTI_AGENT_STAGE_LATENCY{stage}` `MULTI_AGENT_STAGE_ERRORS{stage}`
  - `multi_agent.py` 各阶段套 timer
- **估算**：~30 行

### 4. ingest CLI 进度条
- **目标**：大批量入库时显示进度
- **做法**：`rich.progress.track()` 包 ingest 主循环；显示已处理文件 + chunks
- **估算**：~20 行

### 5. 模型价目表外置
- **目标**：`usage.py` 内硬编码价格不便维护
- **做法**：抽到 `prices.yaml`，启动时加载；支持运行时 `/prices reload`
- **估算**：~40 行 + YAML

---

## 🟡 中价值 / 中成本

### 6. 工具沙箱 Docker 化
- **目标**：`exec_python_snippet` 不应直接跑在主进程
- **做法**：
  - 启动配套 sandbox 容器（python:3.12-slim + 资源限制）
  - 通过 socket / HTTP / docker exec 提交代码
  - 单独工具 `exec_in_sandbox`，原 `exec_python_snippet` 标记 unsafe
- **依赖**：docker-py
- **估算**：~200 行；需 Docker 守护进程

### 7. Qdrant Cloud / 远端模式
- **目标**：embedded 模式不适合多副本部署
- **做法**：
  - 配置加 `QDRANT_URL` / `QDRANT_API_KEY`
  - `vectorstore.py` 按 URL 是否存在切换 embedded / 远端
- **估算**：~30 行

### 8. 文档加载扩展（Excel / PPT / 图片 OCR）
- **目标**：当前不支持 .xlsx .pptx
- **做法**：
  - `openpyxl` 读 Excel（每个 sheet 一个 Document）
  - `python-pptx` 读 PPT（每页一个 Document）
  - `pytesseract` 做扫描版 PDF / 图片 OCR
- **估算**：~120 行 + 较重依赖

### 9. 流式 tool_call 可视化
- **目标**：当前 tool_call_delta 在 agent.py 被丢弃；UI 看不到生成中
- **做法**：CLI / Web 把工具名实时渲染（打字机效果），结束时定型
- **估算**：~60 行

### 10. Self-Query Retriever
- **目标**：LLM 自动从问句抽 metadata filter（如"2025 年的文档"）
- **做法**：
  - 新工具 `search_docs_filtered(query, filter)`
  - system prompt 教 LLM 何时用
  - Qdrant filter 支持时间区间
- **估算**：~150 行

### 11. Reranker ONNX 量化版
- **目标**：BGE Reranker 1.1GB 偏大
- **做法**：换 `BAAI/bge-reranker-v2-m3` ONNX 量化版（~600MB），或 FastEmbed 自带的 reranker
- **依赖**：optimum / fastembed.rerank
- **估算**：~30 行替换

### 12. 工具调用并行化
- **目标**：当前 tool_calls 列表是串行执行
- **做法**：用 `asyncio.gather` 并发跑相互独立的工具调用
- **风险**：file_write 等有副作用工具需互斥
- **估算**：~50 行

### 13. 测试覆盖率徽章 + codecov
- **目标**：CI 上传覆盖率到 codecov.io
- **做法**：
  - `pytest --cov=src --cov-report=xml`
  - GitHub Actions 加 codecov-action
  - README 加 badge
- **估算**：~20 行 YAML

### 14. 配置交叉校验
- **目标**：`rag_top_k_rerank ≤ rag_top_k_retrieval` 等约束
- **做法**：pydantic `@model_validator(mode="after")`
- **估算**：~30 行

### 15. CLI 命令扩展
- `/history`           — 看当前会话所有消息
- `/save <name>`       — 别名当前 thread_id
- `/load <name>`       — 切换到已保存别名
- `/model <name>`      — 运行时换 LLM
- `/last-tool`         — 看上一次工具完整输出（绕过 150 字截断）
- **估算**：~120 行

---

## 🟢 长期 / 探索

### 16. 知识库版本管理
- **目标**：类 git 的 commit / tag 机制，可回滚到历史快照
- **做法**：
  - 每次 ingest 写 snapshot 元数据到 manifest
  - 支持 `python ingest.py rollback <snapshot_id>`
- **估算**：~250 行

### 17. 多模态输入
- **目标**：用户传图片，Agent 看图回答
- **做法**：
  - LLM 协议升级到 vision message 格式
  - CLI / Web 支持图片粘贴 / 上传
- **依赖**：DeepSeek-VL / GPT-4o vision
- **估算**：~200 行

### 18. RAG 调试 / 评估工具
- **目标**：调优分块大小、reranker、top_k 时要客观指标
- **做法**：
  - `eval_rag.py` 读 QA 对（CSV / JSONL）
  - 跑 `search_docs` 与人工标注对比，计 Recall@K / MRR
  - 输出 markdown 报告
- **估算**：~300 行

### 19. 多 Agent 工作流 DSL
- **目标**：当前 multi_agent 是硬编码三角色
- **做法**：YAML 定义 stage + tools + handoff 规则
- **估算**：~400 行

### 20. Prompt 版本管理 + A/B
- **目标**：调整 system prompt 容易，但难量化效果
- **做法**：
  - `prompts/system_v1.md` `system_v2.md`
  - 配置选哪个；自动统计每个版本的 turn 成功率
- **估算**：~150 行

### 21. 长会话摘要压缩
- **目标**：trim_messages 简单粗暴；长对话上下文损失大
- **做法**：
  - 超过 N 条时调 LLM 把旧消息摘要成单条 SystemMessage
  - 类似 LangGraph 的 SummarizationMessage 机制
- **估算**：~120 行

### 22. WebSocket 推送代替 SSE
- **目标**：双向通信支持中断 / 编辑
- **做法**：FastAPI WebSocket endpoint；前端 ws.send / ws.onmessage
- **估算**：~150 行

### 23. 多语言 system prompt
- **目标**：根据用户语言自动切 system prompt
- **做法**：检测首条 user 消息语言，注入对应 prompt
- **估算**：~50 行 + langdetect 依赖

### 24. 工具版本化（OpenAPI schema）
- **目标**：每个工具生成 OpenAPI schema，可被外部系统调用
- **做法**：`@tool` 装饰器额外产出 OpenAPI 片段；FastAPI 自动暴露 `/tools/{name}`
- **估算**：~200 行

### 25. 离线模式
- **目标**：完全断网仍可跑（含 LLM）
- **做法**：
  - 自动检测 Ollama 可用性
  - 自动切到本地小模型 fallback
- **估算**：~80 行

---

## 推荐下一步冲刺 / Suggested Next Sprint

按性价比建议先做 3 件：

1. **#2 Docker Compose 一键部署** — 整套栈打包，演示 / 部署友好
2. **#3 多 Agent 阶段指标** — Grafana 看板已就绪，多 Agent 流水线值得监控
3. **#15 CLI 命令扩展** — 改善日常体验，工作量小

或按主题：
- **可观测性主题**：#1 + #2 + #3 + #13
- **RAG 增强主题**：#8 + #10 + #11 + #18
- **生产化主题**：#2 + #6 + #7 + #14
- **多模态 / 高级 LLM 主题**：#17 + #19 + #21

---

*最后更新 / Last updated: 2026-06-09*
"""
全局配置 / Global configuration.

基于 pydantic-settings 从 .env 与环境变量加载。
Loads from .env + environment variables via pydantic-settings.

为什么用 pydantic-settings / Why pydantic-settings:
    - 自动从 .env / 环境变量读取 / Auto-loads from .env + env vars
    - 类型校验（端口必须是 int，温度在 [0,2] 之间等）
      Type validation (port must be int, temperature in [0,2], etc.)
    - 字段名 snake_case ↔ env 大写自动映射
      Auto-maps snake_case fields ↔ UPPER_SNAKE env vars
    - 模块级单例 → 全应用共享一份配置
      Module-level singleton → app-wide shared config
"""
# Literal 限制字段值枚举（type-check 期 + 运行期）。
# Literal restricts allowed string values (both type-check + runtime).
from typing import Literal

# Field: 字段元数据（默认、范围、说明）/ Field metadata (default, range, description).
from pydantic import Field
# BaseSettings: 配置基类 / SettingsConfigDict: 加载行为配置。
# BaseSettings: config base / SettingsConfigDict: loading behavior.
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # model_config 控制加载行为；不是保存配置值。
    # model_config governs loading; it does NOT store config data.
    model_config = SettingsConfigDict(
        env_file=".env",                # .env 文件路径 / .env file path
        env_file_encoding="utf-8",      # 编码 / encoding
        extra="ignore",                 # 未声明字段忽略（不抛错）/ ignore unknown env vars
    )

    # ─────────── LLM ───────────
    llm_api_key: str                                        # 必填，缺则启动报错 / Required
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    # ge / le：闭区间约束；temperature 仅允许 [0, 2]。
    # ge / le: closed-interval bounds; temperature ∈ [0, 2].
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # gt=0：必须大于 0；max_tokens 不能 0 或负。
    # gt=0: must be > 0; max_tokens can't be 0 / negative.
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_request_timeout: int = Field(default=120, gt=0)     # 秒 / seconds

    # DeepSeek thinking 模式 / DeepSeek thinking mode.
    llm_thinking_enabled: bool = False
    # Literal 让只能填 "high" 或 "max"；其他值启动直接报错。
    # Literal restricts to "high" or "max"; other values fail at startup.
    llm_reasoning_effort: Literal["high", "max"] = "high"
    show_reasoning: bool = True

    # ─────────── Agent ───────────
    agent_recursion_limit: int = Field(default=50, gt=0)
    agent_max_history_messages: int = Field(default=80, gt=0)

    # ─────────── 日志 / Logging ───────────
    log_dir: str = "./logs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_stderr: bool = False        # 默认仅写文件 / File-only by default

    # ─────────── 会话持久化 / Session persistence ───────────
    session_db_path: str = "./sessions.db"

    # ─────────── 文件工具沙箱 / File tools sandbox ───────────
    workspace_dir: str = "./workspace"

    # ─────────── RAG ───────────
    rag_docs_dir: str = "./docs"
    rag_persist_dir: str = "./qdrant_db"
    rag_collection: str = "docs"
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_chunk_size: int = Field(default=512, gt=0)
    rag_chunk_overlap: int = Field(default=64, ge=0)        # ge=0：可以为 0（不重叠）
    rag_top_k_retrieval: int = Field(default=20, gt=0)
    rag_top_k_rerank: int = Field(default=4, gt=0)
    rag_reranker_enabled: bool = True
    rag_reranker_model: str = "BAAI/bge-reranker-base"

    # 混合检索（BM25 + 向量，RRF 融合）/ Hybrid retrieval.
    rag_hybrid_enabled: bool = True
    rag_bm25_top_k: int = Field(default=20, gt=0)
    rag_rrf_k: int = Field(default=60, gt=0)                # RRF 平滑常数


# 模块级单例 / Module-level singleton.
# import 时立即实例化；缺 LLM_API_KEY 时这里就抛错。
# Instantiated at import; missing LLM_API_KEY raises here.
settings = Settings()
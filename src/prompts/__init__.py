"""
提示词模块 / Prompts module.

集中管理所有 LLM 提示词资产：
- 主 Agent 系统提示词（工具列表动态生成）
- 多 Agent 角色提示词（Planner / Researcher / Writer）
- 摘要提示词
- 行为规则（并行白名单等）
- 外部文件加载器

Centralized LLM prompt assets:
- Main Agent system prompt (tool list auto-generated)
- Multi-agent role prompts (Planner / Researcher / Writer)
- Summarization prompts
- Behavioral rules (parallel whitelist, etc.)
- External file loader

设计原则 / Design principles:
    1. 工具描述在 @tool() 装饰器中维护（与函数签名紧耦合）
    2. SYSTEM_PROMPT 中的工具列表通过 build_system_prompt() 动态生成
    3. 行为规则独立为 rules.py，便于审查和迭代
    4. loader.py 支持从外部 .md 文件加载 prompt（记忆/知识注入）
"""
from src.prompts.agent import (
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
    SUMMARY_PREFIX,
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
)
from src.prompts.loader import load_memory_files, load_prompt
from src.prompts.multi_agent import (
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    RESEARCHER_TOOLS,
    WRITER_PROMPT,
)
from src.prompts.rules import (
    BEHAVIOR_RULES,
    BEHAVIOR_RULES_TEXT,
    SAFE_PARALLEL_TOOLS,
)

__all__ = [
    # Agent
    "SYSTEM_PROMPT_TEMPLATE",
    "build_system_prompt",
    "SUMMARIZE_SYSTEM",
    "SUMMARIZE_USER",
    "SUMMARY_PREFIX",
    # Multi-agent
    "PLANNER_PROMPT",
    "RESEARCHER_PROMPT",
    "WRITER_PROMPT",
    "RESEARCHER_TOOLS",
    # Rules
    "SAFE_PARALLEL_TOOLS",
    "BEHAVIOR_RULES",
    "BEHAVIOR_RULES_TEXT",
    # Loader
    "load_prompt",
    "load_memory_files",
]

"""
主 Agent 提示词 / Main Agent prompts.

系统提示词现在通过模板动态生成，工具列表从 TOOLS 注册中心自动同步。
不再需要手动维护 SYSTEM_PROMPT 中的工具枚举。
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────
# 系统提示词模板（{tool_descriptions} 由 build_system_prompt 注入）
# ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with access to multiple tools.

Available tools:
{tool_descriptions}

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


def build_system_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    """
    从 TOOLS 注册中心动态生成系统提示词。
    Build the system prompt dynamically from TOOLS registry.

    工具列表由 @tool 装饰器描述自动生成，与代码始终同步。
    Tool list is auto-generated from @tool descriptions, always in sync.

    Args:
        tool_schemas: TOOLS.schemas() 的返回值 / Return value of TOOLS.schemas()

    Returns:
        完整的系统提示词字符串 / Complete system prompt string.
    """
    lines: list[str] = []
    for s in tool_schemas:
        name = s["function"]["name"]
        desc = s["function"]["description"]
        lines.append(f"- {name}: {desc}")
    tool_descriptions = "\n".join(lines)
    return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)


# ─────────────────────────────────────────────────────────────────────
# 历史摘要提示词 / History summarization prompts
# ─────────────────────────────────────────────────────────────────────
SUMMARIZE_SYSTEM = (
    "Summarize the following conversation history concisely. "
    "Keep key facts, decisions, and context. Output only the summary, no preamble."
)

SUMMARIZE_USER = "Conversation history:\n{history_text}"

SUMMARY_PREFIX = "[Previous conversation summary]\n{text}"

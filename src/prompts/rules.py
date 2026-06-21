"""
行为规则与约束 / Behavioral rules & constraints.

集中定义 Agent 的工具分类白名单和行为约束文本。
这些规则可注入到 system prompt 或用于运行时决策。

Centralized tool classification whitelists and behavioral rules.
Can be injected into system prompt or used for runtime decisions.
"""

# ─────────────────────────────────────────────────────────────────────
# 工具并行化白名单 / Tool parallelization whitelist
#
# 无副作用工具可 asyncio.gather 并行执行。
# Side-effect-free tools safe for asyncio.gather parallel execution.
# ─────────────────────────────────────────────────────────────────────
SAFE_PARALLEL_TOOLS = {
    "search_docs",
    "web_search",
    "web_search_results",
    "calculator",
    "get_current_datetime",
    "read_file",
    "list_directory",
}

# ─────────────────────────────────────────────────────────────────────
# LLM 行为约束 / LLM behavioral constraints
#
# 注入到 system prompt 尾部的显式规则列表。
# Explicit rules appended to the end of the system prompt.
# ─────────────────────────────────────────────────────────────────────
BEHAVIOR_RULES: list[str] = [
    "Never call the same tool with identical arguments twice in a row "
    "without new information; report the issue to the user instead.",
    "Do NOT write wrapper scripts that exec/import existing workspace files "
    "— call run_python_file directly.",
    "Do NOT create permanent files for one-off ad-hoc snippets "
    "— use exec_python_snippet instead.",
    "If a tool fails, explain why and try an alternative approach.",
]

# 单行文本（用于拼接）/ Single-line text for concatenation.
BEHAVIOR_RULES_TEXT: str = "\n".join(f"- {r}" for r in BEHAVIOR_RULES)

"""
提示词加载器 / Prompt loader.

从外部 .md / .txt 文件加载提示词或记忆，支持热重载。
Load prompts or memory from external .md / .txt files with optional hot-reload.

使用场景 / Use cases:
    - 从 docs/ 目录加载领域知识注入 system prompt
    - 从 prompts/ 目录加载自定义角色描述
    - 加载 .claude/memory/ 中的记忆文件
"""
from __future__ import annotations

from pathlib import Path

from src.logging_setup import logger


def load_prompt(path: str | Path) -> str:
    """
    从文件加载提示词文本 / Load prompt text from a file.

    支持 .md / .txt / .prompt 扩展名。
    Supports .md / .txt / .prompt extensions.

    Args:
        path: 文件路径 / File path.

    Returns:
        文件内容（去除首尾空白）/ File content (stripped).

    Raises:
        FileNotFoundError: 文件不存在。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def load_memory_files(directory: str | Path, *, max_files: int = 20) -> list[dict[str, str]]:
    """
    加载记忆目录中的 .md 文件，返回可注入 messages 的 system 消息列表。
    Load .md files from a memory directory as system messages.

    每个文件格式 / Each file format:
        ---
        name: short-slug
        description: one-line summary
        ---
        <content>

    文件内容作为 system role 消息注入到 LLM 上下文。
    File content is injected as system role messages into LLM context.

    Args:
        directory: 记忆目录路径 / Memory directory path.
        max_files: 最多加载文件数 / Max files to load.

    Returns:
        [{"role": "system", "content": "..."}, ...]
    """
    d = Path(directory)
    if not d.is_dir():
        return []

    messages: list[dict[str, str]] = []
    files = sorted(d.glob("*.md"))[:max_files]
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8").strip()
            if not text:
                continue
            # 剥离 frontmatter（--- 块）/ Strip frontmatter block.
            if text.startswith("---"):
                end = text.find("---", 3)
                if end != -1:
                    text = text[end + 3:].strip()
            if text:
                messages.append({
                    "role": "system",
                    "content": f"[Memory: {fp.stem}]\n{text}",
                })
        except Exception:
            logger.warning("Failed to load memory file: {}", fp)

    logger.debug("Loaded {} memory files from {}", len(messages), directory)
    return messages

"""
文件工具（沙箱）/ File tools (sandboxed).

读写仅限 WORKSPACE_DIR 内。越界用 Path.relative_to 严格判定。
Reads/writes are confined to WORKSPACE_DIR. Out-of-bounds checked via
Path.relative_to (avoids the Windows case-insensitive startswith pitfall).

为何用 Path.relative_to 而非 str.startswith / Why `relative_to`:
    Windows 文件系统大小写不敏感时 `str.startswith` 容易误判：
    On case-insensitive Windows, `str.startswith` can be fooled:
        base   = "C:\\Workspace"
        target = "C:\\workspace\\..\\Other"  ← 实际在 base 外，但字符串前缀不同
    Path.relative_to 内部用 PurePath 比较，跨平台行为一致。
    Path.relative_to uses PurePath semantics, consistent across platforms.

术语 / Terminology:
    - sandbox: 沙箱，受限的运行环境 / restricted execution environment
    - resolve(): 把相对路径解析为绝对路径，处理 ../ 等
                 Resolve a path to absolute form, collapsing ../ segments
    - PermissionError: Python 内置的"权限被拒"异常类型
                       Python's built-in "permission denied" exception
"""
from pathlib import Path

from src.config import settings
from src.tools.base import tool


def _resolve(path: str) -> Path:
    """
    解析用户给定的相对路径并校验是否在沙箱内。
    Resolve the user-supplied relative path and ensure it stays in the sandbox.

    步骤 / Steps:
        1. 基准目录绝对化（处理 ./ 与符号链接）
           Absolutise the base dir (resolve ./ and symlinks).
        2. 拼接用户路径并 resolve（这一步会展开 ../，关键）
           Join + resolve — this step expands ../, crucial for safety.
        3. relative_to 失败则越界 → 转 PermissionError 抛出
           If relative_to throws, the path escaped → PermissionError.
    """
    # 基准目录的绝对路径 / Absolute path of the sandbox base.
    base = Path(settings.workspace_dir).resolve()
    # 拼接 + 绝对化；resolve() 会消化掉 ../ 段。
    # Join + resolve; resolve() collapses ../ segments.
    target = (base / path).resolve()
    try:
        # 若 target 不在 base 之下，relative_to 抛 ValueError。
        # If target is not under base, relative_to raises ValueError.
        target.relative_to(base)
    except ValueError:
        # 转成 PermissionError，语义更清晰。
        # Convert to PermissionError for clearer security semantics.
        raise PermissionError(
            f"Access denied: '{path}' is outside workspace '{base}'."
        )
    return target


@tool("Read text content of a file inside the workspace. Input: relative path from workspace root.")
def read_file(file_path: str) -> str:
    # _resolve 抛 PermissionError 时，@tool + safe wrapper 会接住转字符串。
    # If _resolve raises, the tool registry converts to a friendly string.
    target = _resolve(file_path)
    if not target.exists():
        # 文件不存在不视为异常；返回提示给 LLM 阅读。
        # Missing file isn't an exception; LLM can react to the message.
        return f"File not found: {file_path}"
    # 强制 utf-8 防止 Windows 默认编码 (gbk) 把非 ASCII 弄花。
    # Force utf-8 — Windows default (gbk) mangles non-ASCII chars.
    return target.read_text(encoding="utf-8")


@tool("Write text content to a file inside the workspace. Creates parent dirs as needed.")
def write_file(file_path: str, content: str) -> str:
    target = _resolve(file_path)
    # 自动创建父目录，方便 LLM 写嵌套路径（如 "sub/dir/file.txt"）。
    # Auto-create parent dirs so nested paths just work.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    # 返回写入字符数便于 LLM 确认成功 / Char count for confirmation.
    return f"Wrote {len(content)} characters to '{file_path}'."


@tool("List files and subdirectories inside the workspace. Default: workspace root.")
def list_directory(dir_path: str = ".") -> str:
    target = _resolve(dir_path)
    if not target.is_dir():
        return f"Not a directory: {dir_path}"
    # 排序：文件夹在前（按 is_file=False 排前），文件在后；字母序内排。
    # Sort: dirs first (is_file=False), files next; alphabetical within.
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "(empty directory)"
    lines = []
    for entry in entries:
        # "D " 标记目录；文件用空格对齐方便阅读。
        # "D " marks dirs; files use spaces for visual alignment.
        prefix = "  " if entry.is_file() else "D "
        # 文件展示字节大小 / Show byte size for files.
        size = f"  ({entry.stat().st_size}B)" if entry.is_file() else ""
        lines.append(f"{prefix}{entry.name}{size}")
    return "\n".join(lines)
"""
Python 执行工具 / Python execution tools.

提供两个工具 / Two tools:
    run_python_file       — 跑 workspace 内 .py 文件 / Run a workspace .py file
    exec_python_snippet   — 跑临时代码片段 / Run an ad-hoc snippet

安全 / Security:
    - 仅限 workspace 路径 / Workspace-only
    - 30s 默认超时，硬上限 120s / 30s default timeout, 120s hard cap
    - 输出截断 4000 字 / Output truncated to 4000 chars
    - 沙箱模式可选：subprocess / docker / disabled
      Sandbox mode: subprocess / docker / disabled

⚠️ subprocess 仅隔离崩溃；不限网络 / 不限导入；生产用 docker 模式。
⚠️ subprocess isolates crashes only; use docker mode for production.
"""
# 标准库 / Stdlib.
import shutil  # 清理临时目录 / cleanup tmp dir
import uuid  # 临时文件名生成
from pathlib import Path

from src.config import settings
from src.tools.base import tool

# 复用 file_tools 的 _resolve 做沙箱校验 / Reuse sandbox check.
from src.tools.file_tools import _resolve

# 沙箱执行器 / Sandbox executor.
from src.tools.sandbox import run_in_sandbox

# 单个 stream 最大字符 / Per-stream char cap.
_MAX_OUTPUT_CHARS = 4000
# 超时硬上限秒数 / Hard timeout ceiling in seconds.
_TIMEOUT_HARD_CAP = 120
# 临时片段存放子目录（workspace 下）。
# Subdir under workspace for ad-hoc snippets (kept inside sandbox).
_TMP_DIR = ".tmp_exec"


def _truncate(text: str) -> str:
    """超长输出截断 / Truncate oversized output."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    # 截断时附上原长度信息，便于 LLM 知道还有多少没显示。
    # Append remaining char count so LLM knows there's more.
    return text[:_MAX_OUTPUT_CHARS] + f"\n...(truncated {len(text) - _MAX_OUTPUT_CHARS} chars)"


async def _run_sandbox(script: Path, cwd: Path, timeout: int) -> str:
    """
    底层执行：通过 sandbox 模块运行脚本并格式化输出。
    Core runner: delegates to sandbox module, formats output.

    改用 sandbox 模块后支持 docker 模式（网络隔离、资源限制）。
    Now supports docker mode via sandbox module (network isolation, resource caps).
    """
    result = await run_in_sandbox(script, cwd, timeout)

    # 拼装输出：stdout / stderr / exit code 三段。
    # Compose output: stdout / stderr / exit code.
    parts = []
    if result.timed_out:
        parts.append(f"Timeout after {timeout}s.")
    if result.stdout:
        parts.append(f"STDOUT:\n{_truncate(result.stdout)}")
    if result.stderr:
        parts.append(f"STDERR:\n{_truncate(result.stderr)}")
    parts.append(f"Exit code: {result.exit_code}")
    return "\n\n".join(parts)


@tool("Execute a Python file inside the workspace. file_path is relative to workspace root.")
async def run_python_file(file_path: str, timeout_seconds: int = 30) -> str:
    """Execute a Python file in the workspace and return its output."""
    # 沙箱校验：越界路径会抛 PermissionError → 转字符串返回。
    # Sandbox check: out-of-bounds paths raise → string return.
    target = _resolve(file_path)
    if not target.exists():
        return f"File not found: {file_path}"
    # 夹钳超时到合法范围 / Clamp timeout into valid range.
    timeout = max(1, min(timeout_seconds, _TIMEOUT_HARD_CAP))
    # cwd 用脚本所在目录，让脚本能用相对路径访问同目录文件。
    # cwd = script dir, so relative-path file access works.
    return await _run_sandbox(target, target.parent, timeout)


@tool("Execute an ad-hoc Python code snippet. Creates a temp file under workspace, runs, deletes.")
async def exec_python_snippet(code: str, timeout_seconds: int = 30) -> str:
    """Execute an ad-hoc Python snippet."""
    base = Path(settings.workspace_dir).resolve()
    # 临时片段目录；首次执行时自动创建。
    # Temp subdir under workspace; auto-created on first run.
    tmp_dir = base / _TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 唯一文件名（uuid 前 12 位足够避碰撞）。
    # Unique filename (12 hex chars from uuid suffices).
    script = tmp_dir / f"snippet_{uuid.uuid4().hex[:12]}.py"
    timeout = max(1, min(timeout_seconds, _TIMEOUT_HARD_CAP))

    try:
        # 把代码写入临时文件 / Write snippet to disk.
        script.write_text(code, encoding="utf-8")
        # 以 workspace 根目录为 cwd，方便 snippet 引用其他文件。
        # cwd = workspace root so the snippet can reference workspace files.
        return await _run_sandbox(script, base, timeout)
    finally:
        # finally 确保即使中途抛错也清理临时文件。
        # finally ensures cleanup even on exceptions.
        try:
            script.unlink(missing_ok=True)
        except OSError:
            # 文件被锁等极端情况静默忽略。
            # Silently ignore lock-related edge cases.
            pass
        # 临时目录空了也清掉，避免长期堆积空目录。
        # Drop the tmp dir if it ended up empty.
        try:
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                shutil.rmtree(tmp_dir)
        except OSError:
            pass
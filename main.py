#!/usr/bin/env python3
"""
InfiniteLogic 程序入口 / InfiniteLogic entry point.

启动流程 / Bootstrap:
    1. 读 .env → Settings 单例 / Load .env into Settings.
    2. 准备 workspace 目录 / Ensure workspace dir.
    3. 异步打开 SessionStore + 跑 CLI / Open SessionStore + run CLI.

为什么整段 async / Why fully async:
    SessionStore 用 aiosqlite；CLI 用 astream_events 风格异步 IO。
    把入口写成 async 让事件循环唯一，避免跨循环资源问题。
    SessionStore uses aiosqlite; CLI uses async I/O. One event loop
    avoids cross-loop resource issues.
"""
# 标准库 / Stdlib only.
import asyncio
import sys
from pathlib import Path


async def _bootstrap() -> None:
    """
    异步引导流程 / Async bootstrap routine.

    在同一事件循环里完成：配置 → workspace → CLI。
    All inside one event loop: settings → workspace → CLI.
    """
    # 延迟 import：触发 .env 加载；缺 LLM_API_KEY 时这一行就会抛清晰错误。
    # Lazy import: triggers .env load. Missing LLM_API_KEY raises here.
    from src.config import settings

    # 文件工具沙箱目录（read_file / write_file 仅在此目录内操作）。
    # Workspace sandbox; file_tools constrained to this directory.
    Path(settings.workspace_dir).mkdir(parents=True, exist_ok=True)

    # 再延迟 import CLI：让上面的配置/目录就绪后再加载 agent / tools。
    # Lazy-import CLI so settings + workspace are ready first.
    from src.cli import run_cli

    # 进入交互循环；阻塞直到用户 /quit 或 Ctrl+C。
    # Enter interactive loop; blocks until /quit or Ctrl+C.
    await run_cli()


def main() -> None:
    """
    同步入口；负责跑 asyncio 事件循环 + 顶层错误兜底。
    Sync entry; runs the asyncio loop + top-level error handling.
    """
    try:
        # asyncio.run() 创建一个事件循环并跑完 _bootstrap 协程。
        # asyncio.run() creates a loop and runs the coroutine to completion.
        asyncio.run(_bootstrap())
    except KeyboardInterrupt:
        # Ctrl+C 顶层兜底；CLI 内层一般会先捕获，这里防再次冒泡。
        # Top-level Ctrl+C fallback; CLI usually catches first.
        pass
    except Exception as exc:  # noqa: BLE001
        # 初始化异常（缺 API key、模型连接失败等）→ stderr + 退出码 1。
        # Init failures → stderr + exit code 1.
        print(f"Failed to start InfiniteLogic: {exc}", file=sys.stderr)
        print("Check your .env file and ensure LLM_API_KEY is set.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # 仅在直接执行（python main.py）时运行；被 import 时不执行。
    # Runs only on direct execution; skipped when imported.
    main()
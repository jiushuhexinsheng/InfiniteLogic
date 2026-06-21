"""
Python 执行沙箱 / Python execution sandbox.

根据 SANDBOX_MODE 选择执行策略：
- subprocess: 沿用现有子进程方式（仅隔离崩溃）
- docker: Docker 容器隔离（网络/none，只读文件系统，资源限制）
- disabled: 禁用 Python 执行

Execution strategy gated by SANDBOX_MODE:
- subprocess: existing subprocess approach (crash isolation only)
- docker: Docker container (no network, read-only fs, resource caps)
- disabled: execution forbidden

设计 / Design:
    沙箱模块只负责"在受控环境执行一段 Python 脚本并返回输出"。
    不关心脚本从哪里来、怎么生成——由调用方（python_exec.py）负责。
    The sandbox module only cares about "run a script in a controlled env".
    Where the script comes from is the caller's concern (python_exec.py).
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.logging_setup import logger


@dataclass
class SandboxResult:
    """沙箱执行结果 / Sandbox execution result."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


# 输出长度上限 / Output char limit for each stream.
_OUTPUT_LIMIT = 4000
# Docker 容器级超时秒数 / Container-level timeout in seconds.
_DOCKER_TIMEOUT = 30
# subprocess 超时硬上限 / Hard subprocess timeout ceiling.
_SUBPROCESS_HARD_CAP = 120


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    """超长输出截断 / Truncate oversized output with count info."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(truncated {len(text) - limit} chars)"


def _docker_available() -> bool:
    """检查 Docker 是否可用 / Check if Docker is available."""
    return shutil.which("docker") is not None


async def _run_docker(script_path: Path, cwd: Path, timeout: int) -> SandboxResult:
    """
    在 Docker 容器中执行 Python 脚本。
    Run a Python script inside a Docker container.

    安全边界 / Security boundaries:
        --network=none          无网络访问 / No network
        --read-only             只读根文件系统 / Read-only rootfs
        --tmpfs /tmp:size=64M   临时可写空间 / Temp writable space
        --cpus=0.5              半核 CPU / Half-core limit
        --memory=128M            内存上限 / Memory cap
        --pids-limit=50         进程数限制 / Process count limit
        --security-opt=no-new-privileges  禁止提权 / No privilege escalation
    """
    image = settings.sandbox_docker_image
    script_name = script_path.name

    # 把脚本从 workspace 复制到临时目录，再 mount 进容器。
    # Copy script to temp dir for clean mounting.
    tmp_dir = Path(settings.workspace_dir).resolve() / ".sandbox_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / f"{uuid.uuid4().hex[:12]}_{script_name}"
    dst.write_bytes(script_path.read_bytes())

    try:
        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            "--read-only",
            "--tmpfs", "/tmp:size=64M,exec",
            f"--cpus={settings.sandbox_cpu_limit}",
            f"--memory={settings.sandbox_memory_limit}",
            "--pids-limit=50",
            "--security-opt=no-new-privileges",
            "-v", f"{dst.resolve()}:/sandbox/script.py:ro",
            "-w", "/sandbox",
            image,
            "timeout", str(timeout),
            "python", "/sandbox/script.py",
        ]

        logger.debug("docker sandbox: {}", " ".join(cmd))

        # subprocess timeout 作为兜底（docker timeout 是容器内超时）。
        # subprocess timeout as fallback (docker timeout is inside-container).
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + 10
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                stdout="", stderr="Sandbox timed out.",
                exit_code=-1, timed_out=True,
            )

        stdout = _truncate(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = _truncate(stderr_bytes.decode("utf-8", errors="replace"))
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode or 0,
            timed_out=False,
        )
    finally:
        # 清理临时脚本 / Clean up temp script.
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass


async def _run_subprocess(script_path: Path, cwd: Path, timeout: int) -> SandboxResult:
    """
    在子进程中执行 Python 脚本（现有方式，仅隔离崩溃）。
    Run a Python script in a subprocess (current approach, crash-only isolation).
    """
    clamped = max(1, min(timeout, _SUBPROCESS_HARD_CAP))
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=clamped
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return SandboxResult(
                stdout="", stderr=f"Timeout after {clamped}s.",
                exit_code=-1, timed_out=True,
            )

        stdout = _truncate(stdout_bytes.decode("utf-8", errors="replace"))
        stderr = _truncate(stderr_bytes.decode("utf-8", errors="replace"))
        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode or 0,
            timed_out=False,
        )
    except Exception as exc:
        return SandboxResult(
            stdout="", stderr=f"Subprocess error: {exc}",
            exit_code=-1, timed_out=False,
        )


async def run_in_sandbox(
    script_path: Path,
    cwd: Path,
    timeout: int = 30,
) -> SandboxResult:
    """
    在沙箱中执行 Python 脚本 / Execute a Python script in a sandbox.

    Args:
        script_path: 脚本文件绝对路径 / Absolute path to the script
        cwd: 工作目录 / Working directory
        timeout: 执行超时秒数 / Execution timeout in seconds

    Returns:
        SandboxResult with stdout, stderr, exit_code, timed_out.
    """
    mode = settings.sandbox_mode

    if mode == "disabled":
        return SandboxResult(
            stdout="",
            stderr="Python execution is disabled. Set SANDBOX_MODE=subprocess or docker.",
            exit_code=-1,
            timed_out=False,
        )

    if mode == "docker":
        if not _docker_available():
            logger.warning(
                "SANDBOX_MODE=docker but docker not found; falling back to subprocess."
            )
            return await _run_subprocess(script_path, cwd, timeout)
        try:
            return await _run_docker(script_path, cwd, timeout)
        except Exception as exc:
            logger.exception("Docker sandbox failed; falling back to subprocess")
            return SandboxResult(
                stdout="",
                stderr=f"Docker sandbox error (falling back to subprocess): {exc}",
                exit_code=-1,
                timed_out=False,
            )

    # mode == "subprocess"
    return await _run_subprocess(script_path, cwd, timeout)

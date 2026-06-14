"""
日志初始化 / Logging setup.

用 loguru 替代 std logging：单行配置、结构化、按 level 分流。
Uses loguru: single-line config, structured fields, level-based routing.

写入文件 / Files:
    logs/openbase.log         INFO+（全量轮转）
    logs/openbase.error.log   ERROR+（异常专用，含 traceback）

CLI 默认不打日志到 stderr，避免与 rich 输出抢屏。
By default no stderr output to avoid mixing with rich console.

为什么用 loguru 而不是 std logging / Why loguru:
    - 单行 logger.add() 配置完整 sink（轮转 / 格式 / 过滤）
      One-line logger.add() configures entire sink (rotation / format / filter)
    - 内置异常 backtrace 美化 / Built-in pretty traceback
    - format string 支持 {time} {level} 等占位符
      Format string supports {time} {level} placeholders
"""
from pathlib import Path

# loguru 全局 logger 单例；直接 import 即用，无 getLogger 模板代码。
# loguru's global logger singleton; just import and use.
from loguru import logger

from src.config import settings


# 模块级哨兵防止重复初始化（多次 import / 重启场景）。
# Module-level sentinel prevents double-init across re-imports.
_INITIALIZED = False


def setup_logging() -> None:
    """
    初始化日志（幂等）/ Initialize logging (idempotent).

    幂等性 / Idempotency:
        重复调用直接 return；测试 / Streamlit reload 等场景安全。
        Safe under repeated calls (tests / Streamlit reload).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 先移除默认 sink（stderr），让我们完全掌控输出去向。
    # Remove the default stderr sink so we fully control output routing.
    logger.remove()

    # 可选 stderr sink：仅在配置开启时才加。
    # Optional stderr sink; only added if user opted in.
    if settings.log_to_stderr:
        logger.add(
            sink=lambda msg: print(msg, end=""),     # 简单写 stderr / Plain print to stderr
            level=settings.log_level,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:{line} | {message}",
        )

    # 主日志文件：按大小轮转 / Main log file with size rotation.
    logger.add(
        log_dir / "openbase.log",
        level=settings.log_level,
        rotation="10 MB",        # 单文件超过 10MB → 轮转 / Rotate at 10MB
        retention=5,             # 最多保留 5 份 / Keep 5 rotations
        encoding="utf-8",        # 中文兼容 / CJK compatibility
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # 错误专用 sink：仅 ERROR+，附 traceback 便于排查。
    # Error-only sink with traceback for debugging.
    logger.add(
        log_dir / "openbase.error.log",
        level="ERROR",
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        backtrace=True,           # 显示完整异常栈 / Print full backtrace
        diagnose=False,           # diagnose=True 暴露变量值；生产关掉
                                  # diagnose=True leaks variable values; off in prod
    )

    _INITIALIZED = True
    logger.info("Logging initialized: dir={}, level={}", log_dir, settings.log_level)


# 显式 export logger 让其他模块 from src.logging_setup import logger 即用。
# Explicit export so callers can `from src.logging_setup import logger`.
__all__ = ["setup_logging", "logger"]
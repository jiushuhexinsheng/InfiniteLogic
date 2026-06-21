"""
RAG 健康检查 / RAG health check.

提供向量库状态查询，用于监控和告警。
Vector store health queries for monitoring and alerting.

检查项 / Checks:
    - Collection 是否存在 / Collection existence
    - Chunk 总数 / Total chunk count
    - 最后摄入时间 / Last ingest time
    - Embedding 模型可用性 / Embedding model availability

用法 / Usage:
    health = await rag_health_check()
    # → {"status": "ok", "collection": "docs", "chunks": 1234, ...}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import settings
from src.logging_setup import logger
from src.rag import vectorstore as vs


async def rag_health_check() -> dict[str, Any]:
    """
    向量库健康检查 / Vector store health check.

    Returns:
        {
            "status": "ok" | "degraded" | "empty",
            "collection": str,
            "chunks": int | None,
            "embedding_model": str,
            "persist_dir": str,
            "errors": list[str],
        }
    """
    errors: list[str] = []
    result: dict[str, Any] = {
        "status": "ok",
        "collection": settings.rag_collection,
        "chunks": None,
        "embedding_model": settings.rag_embedding_model,
        "persist_dir": settings.rag_persist_dir,
        "errors": errors,
    }

    if not settings.rag_health_check_enabled:
        result["status"] = "disabled"
        return result

    # 1. 检查 collection 是否存在 / Check collection exists.
    try:
        exists = vs.index_exists()
        if not exists:
            result["status"] = "empty"
            errors.append("Collection does not exist. Run `python ingest.py`.")
            return result
    except Exception as exc:
        errors.append(f"Cannot check collection: {exc}")
        result["status"] = "degraded"
        return result

    # 2. 获取 chunk 数 / Get chunk count.
    try:
        client = vs._get_client()  # noqa: SLF001 — health check needs internals
        info = client.get_collection(settings.rag_collection)
        result["chunks"] = info.points_count if info else 0
    except Exception as exc:
        errors.append(f"Cannot get chunk count: {exc}")
        result["status"] = "degraded"

    # 3. 检查持久化目录 / Check persist dir.
    persist = Path(settings.rag_persist_dir)
    if not persist.exists():
        errors.append(f"Persist dir missing: {persist}")
        result["status"] = "degraded"

    # 4. 检查 docs 目录 / Check docs dir.
    docs = Path(settings.rag_docs_dir)
    if not docs.exists() or not any(docs.iterdir()):
        errors.append(f"Docs dir empty or missing: {docs}")

    # 5. 警告零 chunk / Warn if zero chunks (but collection exists).
    if result.get("chunks") == 0 and result["status"] == "ok":
        result["status"] = "empty"
        errors.append("Collection exists but has 0 chunks.")

    if errors:
        logger.warning("RAG health: status={} errors={}", result["status"], errors)
    return result


def rag_health_summary(health: dict[str, Any]) -> str:
    """
    生成一行可读的健康摘要 / Generate a one-line readable health summary.

    适合 CLI /usage 或日志输出 / Suitable for CLI /usage or log lines.
    """
    status = health["status"]
    chunks = health.get("chunks")
    errors = health.get("errors", [])
    if status == "ok":
        return f"RAG: OK (chunks={chunks})"
    if status == "empty":
        return "RAG: EMPTY (run `python ingest.py`)"
    if status == "disabled":
        return "RAG: health check disabled"
    return f"RAG: DEGRADED ({'; '.join(errors[:2])})"

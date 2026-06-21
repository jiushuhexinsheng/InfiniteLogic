"""
RAG 文档监听器 / RAG document watcher.

通过轮询检测 docs 目录文件变更，自动触发 ingest。
Polling-based file change detection; auto-triggers ingest on changes.

设计选择 / Design choice:
    不用 watchdog（省一个依赖），用 asyncio 定时轮询文件 mtime。
    No watchdog dependency — poll file mtime on asyncio timer.

    适合中小规模文档库；大规模 (>10K 文件) 建议换 watchdog 或 inotify。
    Suitable for small/medium doc sets; watchman/inotify better at >10K files.

使用 / Usage:
    watcher = RagWatcher()
    await watcher.start()   # 后台任务 / background task
    ...
    await watcher.stop()    # 停止 / stop
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from src.config import settings
from src.logging_setup import logger
from src.rag.ingest_pipeline import ingest
from src.rag.loader import collect_files

# 轮询间隔秒数 / Poll interval in seconds.
_POLL_INTERVAL = 30
# 去重冷却：同一文件最小变更间隔 / Debounce: min interval between re-ingests.
_DEBOUNCE_SECONDS = 60


class RagWatcher:
    """
    文档目录监听器 / Document directory watcher.

    后台 asyncio 任务定期扫描 docs 目录，检测新增/修改/删除的文件，
    自动调用 ingest pipeline 更新向量库。
    Background asyncio task scans docs dir periodically, detects
    added/modified/deleted files, auto-ingests.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        # 记录文件 hash，用于检测变更 / Track file hashes for change detection.
        self._file_hashes: dict[str, str] = {}
        self._last_ingest: dict[str, float] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """启动后台监听 / Start background watcher."""
        if not settings.rag_auto_ingest:
            logger.info("RAG watcher: auto-ingest disabled (RAG_AUTO_INGEST=false)")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("RAG watcher: started (poll={}s)", _POLL_INTERVAL)

    async def stop(self) -> None:
        """停止后台监听 / Stop background watcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("RAG watcher: stopped")

    async def _loop(self) -> None:
        """后台轮询循环 / Background polling loop."""
        while self._running:
            try:
                await self._scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("RAG watcher: scan error")
            await asyncio.sleep(_POLL_INTERVAL)

    async def _scan(self) -> None:
        """扫描文档目录，检测变更 / Scan docs dir for changes."""
        docs_dir = Path(settings.rag_docs_dir)
        if not docs_dir.exists():
            return

        files = collect_files(docs_dir)
        if not files:
            return

        # 计算所有文件 hash / Compute hashes for all files.
        current: dict[str, str] = {}
        for fp in files:
            try:
                h = _file_hash(fp)
                current[str(fp)] = h
            except Exception:
                logger.warning("RAG watcher: cannot hash {}", fp)

        # 检测变更 / Detect changes.
        new_files = [p for p in current if p not in self._file_hashes]
        modified = [
            p for p in current
            if p in self._file_hashes and current[p] != self._file_hashes[p]
        ]
        deleted = [p for p in self._file_hashes if p not in current]

        changed = new_files + modified + deleted
        if not changed:
            return

        # 去重冷却 / Debounce check.
        now = asyncio.get_event_loop().time()
        eligible = [
            p for p in changed
            if p not in self._last_ingest or now - self._last_ingest[p] > _DEBOUNCE_SECONDS
        ]
        if not eligible:
            return

        logger.info(
            "RAG watcher: detected changes — new={} mod={} del={}",
            len(new_files), len(modified), len(deleted),
        )

        # 触发入库 / Trigger ingest.
        try:
            eligible_paths = [Path(p) for p in eligible if Path(p).exists()]
            if eligible_paths:
                stats = ingest(eligible_paths, clear=False)
                logger.info(
                    "RAG watcher: auto-ingested — new={} updated={} chunks={} skipped={}",
                    stats["new_files"], stats["updated"],
                    stats["new_chunks"], stats["skipped"],
                )
        except Exception:
            logger.exception("RAG watcher: auto-ingest failed")

        # 更新状态 / Update state.
        for p in eligible:
            self._last_ingest[p] = now
        self._file_hashes = current


def _file_hash(path: Path) -> str:
    """快速文件 hash（SHA256 前 16 字符）/ Quick file hash (first 16 chars of SHA256)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # 只读前 64KB + 文件大小（对大文件友好）。
        # Read first 64KB + file size (friendly to large files).
        chunk = f.read(65536)
        hasher.update(chunk)
        hasher.update(str(path.stat().st_size).encode())
    return hasher.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────
# 全局单例 / Global singleton
# ─────────────────────────────────────────────────────────────────────
_watcher: RagWatcher | None = None


def get_rag_watcher() -> RagWatcher:
    """获取全局 RagWatcher 单例 / Get global RagWatcher singleton."""
    global _watcher
    if _watcher is None:
        _watcher = RagWatcher()
    return _watcher

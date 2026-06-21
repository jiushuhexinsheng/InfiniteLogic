"""
会话持久化 / Session persistence.

把 (thread_id, message) 序列写入 SQLite，重启可恢复。
Stores (thread_id, message) tuples in SQLite for cross-process recovery.

表结构 / Schema:
    messages(
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id  TEXT    NOT NULL,
        idx        INTEGER NOT NULL,         -- 顺序号 / order within thread
        payload    TEXT    NOT NULL,         -- JSON-encoded message dict
        created_at REAL    NOT NULL          -- UNIX timestamp
    )
    CREATE INDEX ix_messages_thread ON messages(thread_id, idx)

每条消息原样存 JSON：role / content / tool_calls / tool_call_id / reasoning_content。
Each message stored verbatim as JSON: role, content, tool_calls,
tool_call_id, reasoning_content.

生产化改进 / Production improvements (P1):
    - WAL 模式：并发读不阻塞写 / WAL: concurrent reads don't block writes
    - 分页加载：避免长会话内存爆炸 / Paginated load for long sessions
    - 批量提交：每 turn 一次 commit / Batch commit per turn
    - 会话 TTL：自动清理过期会话 / Auto-expire stale sessions
    - 按范围删除：支持历史压缩 / Range delete for history summarization
"""
from __future__ import annotations

# 标准库 / Stdlib.
import json
import time
from pathlib import Path
from typing import Any

# aiosqlite: SQLite 的异步包装；底层仍是 stdlib sqlite3 + threading。
# aiosqlite: async wrapper around stdlib sqlite3 + threading.
import aiosqlite

from src.config import settings
from src.logging_setup import logger

# 建表 + 索引的 SQL；executescript 一次跑多条语句。
# CREATE statements; executescript runs multiple at once.
# WAL 模式通过 PRAGMA journal_mode=WAL 启用。
_INIT_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT    NOT NULL,
    idx        INTEGER NOT NULL,
    payload    TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_thread ON messages(thread_id, idx);
"""


class SessionStore:
    """
    异步会话存储 / Async session store.

    生命周期 / Lifecycle:
        store = await SessionStore.open()   # 打开 + 建表
        ...
        await store.close()                  # 关闭连接

    线程安全 / Thread safety:
        aiosqlite 内部用单线程跑 sqlite，async 调用串行化；
        多协程并发安全，但跨进程仍需注意 SQLite 文件锁。
        aiosqlite serializes calls; multi-coroutine safe.
        Cross-process: rely on SQLite's file lock.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        # 通过工厂方法 .open() 注入连接；不直接对外暴露构造。
        # Constructor takes the connection; users call .open() factory.
        self._conn = conn

    @classmethod
    async def open(cls) -> SessionStore:
        """打开 SQLite 连接 + 建表 + WAL / Open + init schema + WAL."""
        db_path = Path(settings.session_db_path)
        # 父目录可能不存在，比如默认 ./sessions.db 在仓根。
        # Parent dir may not exist; create as needed.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        # executescript 跑 PRAGMA WAL + CREATE TABLE + CREATE INDEX。
        # executescript runs WAL pragma + schema DDL in one shot.
        await conn.executescript(_INIT_SQL)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        """关闭底层连接 / Close the underlying connection."""
        await self._conn.close()

    # ──────────────────────────────────────────────────────
    # 读 / Read
    # ──────────────────────────────────────────────────────
    async def load_messages(
        self,
        thread_id: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        加载会话消息（支持分页）/ Load thread messages with optional pagination.

        offset / limit 为 0 / None 时加载全部（向后兼容）。
        offset=0, limit=None loads all (backward compatible).
        """
        if limit:
            async with self._conn.execute(
                "SELECT payload FROM messages WHERE thread_id = ? "
                "ORDER BY idx ASC LIMIT ? OFFSET ?",
                (thread_id, limit, offset),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self._conn.execute(
                "SELECT payload FROM messages WHERE thread_id = ? ORDER BY idx ASC",
                (thread_id,),
            ) as cur:
                rows = await cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    async def load_messages_paginated(
        self,
        thread_id: str,
        page: int = 0,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        按页加载消息 / Load messages by page number.

        page=0 是最新一页（倒数），page=1 是前一页，以此类推。
        page=0 is the most recent page, page=1 the previous, etc.

        用于长会话中只加载最近的消息，避免内存爆炸。
        Use for long sessions to avoid loading all messages into memory.
        """
        size = page_size or settings.session_message_page_size
        # 先拿到该 thread 总消息数 / Get total count first.
        async with self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
            (thread_id,),
        ) as cur:
            row = await cur.fetchone()
            total = row[0] if row else 0

        if total == 0:
            return []

        # 从后往前分页 / Paginate from the end.
        # page=0 → 最后 size 条；page=1 → 倒数 size*2 到 size 条。
        offset = max(0, total - (page + 1) * size)
        limit = size
        if page == 0:
            # 最新页可能不足一页 / Last page may be partial.
            limit = total - offset

        return await self.load_messages(thread_id, offset=offset, limit=limit)

    async def count_messages(self, thread_id: str) -> int:
        """获取会话消息总数 / Get message count for a thread."""
        async with self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
            (thread_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    async def _next_index(self, thread_id: str) -> int:
        """
        下一条消息的顺序号 / Next idx for this thread.

        COALESCE(MAX(idx), -1) + 1：MAX 为 NULL（无消息）时返回 0。
        COALESCE handles the "empty thread" case → start at 0.
        """
        async with self._conn.execute(
            "SELECT COALESCE(MAX(idx), -1) + 1 FROM messages WHERE thread_id = ?",
            (thread_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ──────────────────────────────────────────────────────
    # 写 / Write
    # ──────────────────────────────────────────────────────
    async def append(self, thread_id: str, message: dict[str, Any]) -> None:
        """追加单条消息 / Append one message."""
        idx = await self._next_index(thread_id)
        # ensure_ascii=False 保留中文原文，便于直接看 sqlite browser 检查。
        # ensure_ascii=False keeps CJK readable in DB browsers.
        payload = json.dumps(message, ensure_ascii=False)
        await self._conn.execute(
            "INSERT INTO messages (thread_id, idx, payload, created_at) VALUES (?, ?, ?, ?)",
            (thread_id, idx, payload, time.time()),
        )
        await self._conn.commit()

    async def append_many(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        """批量追加 / Bulk append (single commit for speed)."""
        if not messages:
            return
        start = await self._next_index(thread_id)
        now = time.time()
        # 构造批量 rows / Build batch rows.
        rows = [
            (thread_id, start + i, json.dumps(m, ensure_ascii=False), now)
            for i, m in enumerate(messages)
        ]
        # executemany 一次插入多条；比 N 次 execute 快得多。
        # executemany inserts N rows in one go; much faster than N executes.
        await self._conn.executemany(
            "INSERT INTO messages (thread_id, idx, payload, created_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        await self._conn.commit()

    # ──────────────────────────────────────────────────────
    # 删除 / Delete
    # ──────────────────────────────────────────────────────
    async def clear(self, thread_id: str) -> None:
        """删除某会话 / Drop a thread's history."""
        await self._conn.execute(
            "DELETE FROM messages WHERE thread_id = ?", (thread_id,)
        )
        await self._conn.commit()

    async def delete_old_messages(
        self, thread_id: str, keep_recent: int
    ) -> int:
        """
        删除旧消息，仅保留最近 N 条（按 idx 降序）。
        Delete old messages, keeping only the most recent N by idx.

        返回删除的条数 / Returns number of deleted messages.

        用于历史压缩：LLM 生成摘要后删除旧消息。
        Used with history summarization: delete old msgs after summarization.
        """
        # 先找到保留的起始 idx / Find the cutoff idx.
        async with self._conn.execute(
            "SELECT idx FROM messages WHERE thread_id = ? ORDER BY idx DESC LIMIT 1 OFFSET ?",
            (thread_id, keep_recent - 1),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return 0  # 不足 keep_recent 条，不删 / Less than keep_recent, skip.

        cutoff = row[0]
        async with self._conn.execute(
            "DELETE FROM messages WHERE thread_id = ? AND idx < ?",
            (thread_id, cutoff),
        ) as cur:
            deleted = cur.rowcount
        await self._conn.commit()
        return deleted

    async def cleanup_expired(self) -> int:
        """
        清理过期会话 / Clean up expired sessions.

        TTL 由 session_ttl_days 控制；0 表示永不过期。
        TTL controlled by session_ttl_days; 0 = never expire.

        返回清理的会话数 / Returns number of threads cleaned.
        """
        ttl_days = settings.session_ttl_days
        if ttl_days <= 0:
            return 0

        cutoff = time.time() - (ttl_days * 86400)
        # 找到最后活动时间早于 cutoff 的 thread_id。
        # Find threads whose last activity is before cutoff.
        async with self._conn.execute(
            "SELECT thread_id FROM messages GROUP BY thread_id "
            "HAVING MAX(created_at) < ?",
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()

        expired = [r[0] for r in rows]
        if not expired:
            return 0

        for tid in expired:
            await self._conn.execute(
                "DELETE FROM messages WHERE thread_id = ?", (tid,)
            )
        await self._conn.commit()
        logger.info("Session TTL: cleaned {} expired threads", len(expired))
        return len(expired)
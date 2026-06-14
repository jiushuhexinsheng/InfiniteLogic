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

为什么 SQLite 而非 Redis / Why SQLite over Redis:
    - 零运维（单文件）/ Zero ops (single file)
    - aiosqlite 完全异步 / Fully async via aiosqlite
    - 中小规模够用；大规模换 Postgres
      Good enough for small/medium; switch to Postgres at scale
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


# 建表 + 索引的 SQL；executescript 一次跑多条语句。
# CREATE statements; executescript runs multiple at once.
_INIT_SQL = """
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
    async def open(cls) -> "SessionStore":
        """打开 SQLite 连接 + 建表 / Open + init schema."""
        db_path = Path(settings.session_db_path)
        # 父目录可能不存在，比如默认 ./sessions.db 在仓根。
        # Parent dir may not exist; create as needed.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        # executescript 跑 CREATE TABLE + CREATE INDEX 两条。
        # executescript runs CREATE TABLE + CREATE INDEX in one shot.
        await conn.executescript(_INIT_SQL)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        """关闭底层连接 / Close the underlying connection."""
        await self._conn.close()

    # ──────────────────────────────────────────────────────
    # 读 / Read
    # ──────────────────────────────────────────────────────
    async def load_messages(self, thread_id: str) -> list[dict[str, Any]]:
        """加载某会话全部消息（按顺序）/ Load all messages of a thread in order."""
        # 用参数化查询（?）防 SQL 注入。
        # Parameterized query (?) prevents SQL injection.
        async with self._conn.execute(
            "SELECT payload FROM messages WHERE thread_id = ? ORDER BY idx ASC",
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
        # rows 是 [(payload,), ...]；用 r[0] 拿字段 0，json.loads 反序列化。
        # rows is [(payload,), ...]; r[0] is the payload, decode JSON.
        return [json.loads(r[0]) for r in rows]

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
        # 每次 append 立即 commit，保证 crash 后不丢消息。
        # Commit per append; ensures durability across crashes.
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

    async def clear(self, thread_id: str) -> None:
        """删除某会话 / Drop a thread's history."""
        await self._conn.execute(
            "DELETE FROM messages WHERE thread_id = ?", (thread_id,)
        )
        await self._conn.commit()
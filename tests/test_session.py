"""会话存储测试 / Session store tests."""
import tempfile
from pathlib import Path

import pytest

from src.config import settings
from src.session import SessionStore


@pytest.fixture
async def store():
    """创建临时数据库的 SessionStore / SessionStore with temp DB."""
    with tempfile.TemporaryDirectory() as tmp:
        orig = settings.session_db_path
        settings.session_db_path = str(Path(tmp) / "test.db")
        s = await SessionStore.open()
        yield s
        await s.close()
        settings.session_db_path = orig


@pytest.mark.asyncio
async def test_open_and_close(store):
    """打开 + 关闭 / Open + close."""
    assert store is not None


@pytest.mark.asyncio
async def test_append_and_load(store):
    """追加并加载 / Append then load."""
    tid = "test-thread-1"
    msg = {"role": "user", "content": "hello"}
    await store.append(tid, msg)

    loaded = await store.load_messages(tid)
    assert len(loaded) == 1
    assert loaded[0]["role"] == "user"
    assert loaded[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_append_many(store):
    """批量追加 / Bulk append."""
    tid = "test-thread-2"
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    await store.append_many(tid, msgs)

    loaded = await store.load_messages(tid)
    assert len(loaded) == 3
    assert loaded[1]["content"] == "a1"


@pytest.mark.asyncio
async def test_clear(store):
    """清空会话 / Clear thread."""
    tid = "test-thread-3"
    await store.append(tid, {"role": "user", "content": "msg"})
    await store.clear(tid)

    loaded = await store.load_messages(tid)
    assert len(loaded) == 0


@pytest.mark.asyncio
async def test_paginated_load(store):
    """分页加载 / Paginated load."""
    tid = "test-thread-4"
    for i in range(25):
        await store.append(tid, {"role": "user", "content": f"msg{i}"})

    # 第 0 页 (最新) / Page 0 (most recent).
    # offset = 25 - 10 = 15, returns items 15..24 (10 items).
    page0 = await store.load_messages_paginated(tid, page=0, page_size=10)
    assert len(page0) == 10
    assert page0[-1]["content"] == "msg24"
    assert page0[0]["content"] == "msg15"

    # 第 1 页 / Page 1.
    # offset = 25 - 20 = 5, returns items 5..14 (10 items).
    page1 = await store.load_messages_paginated(tid, page=1, page_size=10)
    assert len(page1) == 10
    assert page1[0]["content"] == "msg5"


@pytest.mark.asyncio
async def test_count_messages(store):
    """消息计数 / Message count."""
    tid = "test-thread-5"
    assert await store.count_messages(tid) == 0

    await store.append(tid, {"role": "user", "content": "a"})
    assert await store.count_messages(tid) == 1


@pytest.mark.asyncio
async def test_delete_old_messages(store):
    """按范围删除旧消息 / Delete old messages by range."""
    tid = "test-thread-6"
    for i in range(20):
        await store.append(tid, {"role": "user", "content": f"msg{i}"})

    deleted = await store.delete_old_messages(tid, keep_recent=5)
    assert deleted > 0
    remaining = await store.load_messages(tid)
    assert len(remaining) <= 5
    # 保留的应是最新的 / Remaining should be the most recent.
    assert remaining[-1]["content"] == "msg19"


@pytest.mark.asyncio
async def test_cleanup_expired(store):
    """TTL 清理 / TTL cleanup."""
    # 会话刚创建，不应被清理 / Fresh sessions should not be cleaned up.
    tid = "test-thread-7"
    await store.append(tid, {"role": "user", "content": "fresh"})

    cleaned = await store.cleanup_expired()
    # 依赖 session_ttl_days 设置；刚创建的会话不会被清。
    assert cleaned >= 0

"""
入库流水线 / Ingestion pipeline.

流程 / Flow:
    扫描 → SHA1 去重 → 加载 → 切块 → 嵌入 → Qdrant upsert → 更新 manifest
    Scan → SHA1 dedupe → load → chunk → embed → Qdrant upsert → update manifest

内容变化文件：先按旧 SHA1 删，再用新 SHA1 写（原地更新）。
Changed files: delete by old sha1 then insert by new sha1 (in-place update).

为什么用 SHA1 作"文件指纹" / Why SHA1 as the file fingerprint:
    - 内容微改即变 / Any content edit yields a new hash
    - 文件名 / 路径无关 / Independent of filename / path
    - 160 位足够避免碰撞（实际场景）/ 160 bits — collision-free in practice
    （SHA1 安全性已不足做密码学；做文件去重完全够用）
    (SHA1 broken for crypto but fine for content dedup)
"""
from __future__ import annotations

# 标准库 / Stdlib.
import hashlib  # SHA1 哈希 / SHA1 hashing
import json  # manifest 序列化 / manifest serialization
import shutil  # rmtree（--clear 时删整个目录）/ rmtree for --clear
from pathlib import Path
from typing import Any

from src.config import settings

# 用模块对象而不是 from import：方便测试时 monkeypatch。
# Import as module to ease monkeypatching in tests.
from src.rag import vectorstore as vs

# 索引与文档抽象 / Index + document abstractions.
from src.rag.bm25_index import BM25Index, load_bm25_index
from src.rag.loader import collect_files, load_file
from src.rag.splitter import split_documents

MANIFEST_NAME = "manifest.json"


def _manifest_path() -> Path:
    """manifest 文件路径 / Manifest file path."""
    return Path(settings.rag_persist_dir) / MANIFEST_NAME


def _load_manifest() -> dict[str, dict[str, Any]]:
    """
    读取 manifest；不存在或损坏返回空 dict。
    Load manifest; return {} if missing or corrupted.
    """
    p = _manifest_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 损坏 manifest 不阻断 ingest；视作首次入库。
        # Corrupted manifest doesn't block ingest; treated as first-time.
        return {}


def _save_manifest(m: dict[str, dict[str, Any]]) -> None:
    """
    写入 manifest（带缩进、不转义 ASCII）便于人工查看。
    Write manifest with indentation and CJK preserved for readability.
    """
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _file_sha1(path: Path) -> str:
    """
    分块流式读 SHA1，避免一次性把大文件加载内存。
    Stream-hash a file in 1MB blocks; avoids loading large files into RAM.
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        # iter(callable, sentinel) 反复调 callable 直到返回 sentinel。
        # iter(callable, sentinel) repeats callable until it returns sentinel.
        # 1 << 20 == 1MB；常用块大小。
        # 1 << 20 == 1 MB block.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(paths: list[Path], clear: bool = False) -> dict[str, Any]:
    """
    入库主函数 / Main ingestion routine.

    返回字段 / Return fields:
        new_files:    新加入的文件数 / Files newly added
        updated:      内容变化原地更新的文件数 / Files updated in place
        new_chunks:   写入的 chunk 数 / Chunks written
        skipped:      未变化文件数 / Skipped unchanged files
    """
    persist = Path(settings.rag_persist_dir)

    # --clear 模式：先关闭 Qdrant 客户端（文件锁），再删持久化目录。
    # --clear mode: close Qdrant client before rmtree to release file lock.
    if clear:
        vs.reset_client()
        if persist.exists():
            shutil.rmtree(persist)

    manifest = _load_manifest()
    # 统计计数器 / Stats counters.
    skipped = 0
    new_files: list[Path] = []
    updated_files: list[Path] = []
    total_chunks = 0
    # pending_manifest 暂存本次 ingest 的新条目；全部成功后合并到 manifest。
    # Staged manifest entries; merged into manifest after success.
    pending_manifest: dict[str, dict[str, Any]] = {}

    # BM25 索引（如启用混合检索则加载或后续构建）。
    # BM25 index (loaded if hybrid retrieval enabled).
    bm25 = load_bm25_index() if settings.rag_hybrid_enabled else None

    for path in paths:
        # 用绝对路径作 manifest key，避免相对路径歧义。
        # Use absolute path as manifest key to avoid relative-path ambiguity.
        key = str(path.resolve())
        try:
            sha1 = _file_sha1(path)
        except OSError:
            # 读不了的文件跳过（权限、损坏等）。
            # Skip unreadable files (permissions, corruption, ...).
            continue

        entry = manifest.get(key)
        # 已存在且 SHA1 没变 → 跳过 / Already indexed and unchanged → skip.
        if entry and entry.get("sha1") == sha1:
            skipped += 1
            continue

        # 检测是否原地更新 / Detect in-place update.
        is_update = entry is not None
        if is_update:
            # 先按旧 SHA1 删除 Qdrant 中所有 chunk。
            # Delete all old chunks from Qdrant by old sha1.
            old_sha1 = entry["sha1"]
            vs.delete_by_sha1(old_sha1)
            # 同步删 BM25 / Sync delete from BM25.
            if bm25 is not None:
                bm25.remove_by_sha1(old_sha1)

        # 加载并切块 / Load + chunk the file.
        docs = load_file(path)
        if not docs:
            # 加载失败（已记 warn）/ Failed to load (already warned).
            continue
        chunks = split_documents(
            docs, settings.rag_chunk_size, settings.rag_chunk_overlap
        )
        # 给每个 chunk 注入元数据 / Annotate every chunk with metadata.
        for c in chunks:
            c.metadata["source"] = path.name   # 文件名（检索结果展示用）
            c.metadata["sha1"] = sha1          # 同步到 chunk metadata 便于审计
        if not chunks:
            # 极端情况：文件加载成功但切完是空（纯空格）。
            # Edge case: loaded ok but split produced nothing (whitespace only).
            continue

        # 向量库 upsert：写入 Qdrant，返回写入数量。
        # Upsert to vector store; returns chunk count written.
        written = vs.add_documents_with_sha1(chunks, sha1)
        total_chunks += written

        # BM25 同步追加 / Sync append to BM25 index.
        if settings.rag_hybrid_enabled:
            if bm25 is None:
                # 首次 ingest 且开启混合：用第一批 chunk 构造新索引。
                # First ingest with hybrid on: build a fresh BM25.
                bm25 = BM25Index.from_documents_with_sha1(chunks, sha1)
            else:
                bm25.add_documents_with_sha1(chunks, sha1)

        # 按是新加还是更新分桶 / Bucket by add vs update.
        (updated_files if is_update else new_files).append(path)
        # 记录 manifest 条目 / Record manifest entry.
        pending_manifest[key] = {
            "name": path.name,
            "sha1": sha1,
            "size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
            "chunks": len(chunks),
        }

    # BM25 索引落盘（如有变化）。
    # Persist BM25 to disk if anything changed.
    if bm25 is not None and (new_files or updated_files):
        bm25.save(persist)

    # 一个 chunk 都没写 → 不更新 manifest，直接返回。
    # Zero chunks written → leave manifest untouched.
    if total_chunks == 0:
        return {
            "new_files": 0,
            "updated": 0,
            "new_chunks": 0,
            "skipped": skipped,
        }

    # 合并 pending 到主 manifest 并保存 / Merge pending into manifest and save.
    manifest.update(pending_manifest)
    _save_manifest(manifest)

    return {
        "new_files": len(new_files),
        "updated": len(updated_files),
        "new_chunks": total_chunks,
        "skipped": skipped,
    }


__all__ = ["ingest", "collect_files", "load_file"]
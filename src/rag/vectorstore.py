"""
向量库（Qdrant embedded + FastEmbed）/ Vector store using Qdrant + FastEmbed.

为何换 Qdrant / Why Qdrant:
    - 支持按 metadata 过滤删除 / Supports metadata-based deletion
    - 索引随 upsert 持久化 / Persisted to disk transparently
    - 单文件嵌入式（无需起服务）/ Embedded mode (no daemon)

ID 策略 / ID strategy:
    每个 chunk 用 uuid5(sha1 + chunk_idx) 生成稳定 ID。
    Each chunk gets a stable ID via uuid5(sha1 + chunk_idx).
    再次入库同一 chunk 触发幂等 upsert（自动覆盖）。
    Re-ingesting the same chunk produces an idempotent upsert.

术语 / Terminology:
    - Qdrant: Rust 实现的高性能向量数据库 / Rust-based high-perf vector DB
    - Collection: 一组同维度向量的逻辑分组 / A logical group of same-dim vectors
    - Payload: 存在向量旁的任意 JSON 元数据 / Arbitrary JSON metadata stored with vectors
    - UUID5: 基于命名空间 + 名称的确定性 UUID / Deterministic UUID from namespace+name
"""
from __future__ import annotations

# 标准库 uuid 用于生成稳定 point id。
# Stdlib uuid; we use uuid5 for deterministic IDs.
import uuid
from pathlib import Path
from typing import Iterable, Optional

# numpy: FastEmbed 返回 ndarray；Qdrant 消费 ndarray 或 list。
# FastEmbed returns ndarray; Qdrant client accepts ndarray or list of floats.
import numpy as np
# FastEmbed: ONNX Runtime 上跑 embedding；无 PyTorch 重依赖。
# FastEmbed runs ONNX Runtime; avoids the heavy PyTorch dependency.
from fastembed import TextEmbedding
# qdrant-client: 官方 Python SDK；同时支持本地 embedded 和远端服务。
# Official Python SDK; works both embedded and remote.
from qdrant_client import QdrantClient
# qm 是 models 模块的别名，含 PointStruct / Filter / FieldCondition 等。
# qm aliases the `models` module: PointStruct / Filter / FieldCondition / ...
from qdrant_client.http import models as qm

from src.config import settings
from src.rag.document import Document

# 命名空间 UUID，用来生成稳定的 chunk id（同一字符串永远产出同一 UUID）。
# Namespace UUID used to derive stable chunk IDs deterministically.
# 这个常量随便挑一个 UUIDv4 即可，但一旦上线就不要再改（改了所有 id 就变）。
# Any random UUIDv4 works, but never change it once deployed.
_NAMESPACE = uuid.UUID("a0e3a1aa-0000-0000-0000-000000000001")

# Embedding 模型单例 / Single embedding model instance.
# 首次调用 _get_embedder() 时构造；模型权重加载约 1-2 秒。
# Built on first call; model loading takes ~1-2 seconds.
_embedder: TextEmbedding | None = None

# Qdrant 客户端单例 / Qdrant client singleton.
# 注意：embedded 模式不允许同一目录多次实例化（文件锁）。
# Note: embedded mode forbids concurrent init on the same dir (file lock).
_client: QdrantClient | None = None
# 维度缓存：探测一次后保留 / Cached embedding dim after probing once.
_dim: int | None = None


def _get_embedder() -> TextEmbedding:
    """
    懒加载 embedding 模型 / Lazy-load the embedding model.

    用模块级变量代替 lru_cache：FastEmbedEmbeddings 不可哈希。
    Module-level cache (lru_cache won't work — TextEmbedding isn't hashable).
    """
    global _embedder
    if _embedder is None:
        # 首次构造时从 ~/.cache/fastembed 读取或下载 ONNX 模型。
        # On first init, model ONNX is loaded from ~/.cache/fastembed (download if missing).
        _embedder = TextEmbedding(model_name=settings.rag_embedding_model)
    return _embedder


def _embed(texts: list[str]) -> np.ndarray:
    """
    嵌入一组文本，返回 float32 二维矩阵 (N, dim)。
    Embed a batch of texts; returns a float32 (N, dim) ndarray.

    FastEmbed.embed() 是 generator，需要 list 化。
    FastEmbed.embed() returns a generator; consume into a list.
    """
    embedder = _get_embedder()
    # list(generator) 触发实际推理 / list() drives the generator (real inference).
    arrays = list(embedder.embed(texts))
    # 显式指定 dtype；Qdrant 期望 float32。
    # Explicit dtype — Qdrant wants float32.
    return np.asarray(arrays, dtype="float32")


def _detect_dim() -> int:
    """
    探测嵌入维度（首次调用触发，缓存结果）。
    Detect embedding dimensionality once and cache.

    用一个无意义 token 探测，避免错误关联到真实数据。
    Use a sentinel string so it doesn't pollute logs.
    """
    global _dim
    if _dim is None:
        # embed 返回 (1, dim)；取 shape[1] 拿到维度。
        # embed returns (1, dim); shape[1] gives the dim.
        _dim = int(_embed(["__probe__"]).shape[1])
    return _dim


def _stable_point_id(sha1: str, idx: int) -> str:
    """
    用 sha1 + chunk_idx 生成稳定 UUID 作为 Qdrant point id。
    Derive a stable UUID from sha1 + chunk_idx.

    uuid5(namespace, name) 是确定性的：相同输入必产出相同 UUID。
    uuid5(namespace, name) is deterministic — same input → same UUID.
    这保证：同一文件未修改时再次 ingest 不会重复插入（upsert 自动覆盖）。
    Guarantees idempotent re-ingest of unchanged files.
    """
    return str(uuid.uuid5(_NAMESPACE, f"{sha1}:{idx}"))


def _get_client() -> QdrantClient:
    """
    懒加载 Qdrant 客户端 / Lazy-init Qdrant client.

    embedded 模式：传 path 即用本地文件存储，无网络、无守护进程。
    Embedded mode: pass `path` to use local files; no daemon needed.
    """
    global _client
    if _client is None:
        persist = Path(settings.rag_persist_dir)
        persist.mkdir(parents=True, exist_ok=True)
        # QdrantClient(path=...) 触发本地存储；也可改成 url=... 用远端。
        # path=... uses local storage. Use url=... for remote Qdrant Cloud.
        _client = QdrantClient(path=str(persist))
    return _client


def _ensure_collection() -> None:
    """
    确保 collection 存在（不存在则建）。
    Ensure the collection exists; create if missing.
    """
    client = _get_client()
    coll = settings.rag_collection
    # 列出已有 collection；set 推导节省遍历开销。
    # List existing collections; use a set for O(1) lookup.
    existing = {c.name for c in client.get_collections().collections}
    if coll in existing:
        return
    # 维度 = embedding 模型维度（首次会触发探测）。
    # distance 用 COSINE：与 BGE 系列模型推荐一致。
    # Vector dim = embedding model dim. COSINE distance matches BGE conventions.
    client.create_collection(
        collection_name=coll,
        vectors_config=qm.VectorParams(
            size=_detect_dim(),
            distance=qm.Distance.COSINE,
        ),
    )
    # 给 sha1 字段建 KEYWORD payload index，加速 filter 删除。
    # 没有 index 也能工作，但全表扫；KEYWORD 走倒排索引快得多。
    # Without an index Qdrant scans all points; KEYWORD index uses inverted lookup.
    client.create_payload_index(
        collection_name=coll,
        field_name="sha1",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )


# ────────────────────────────────────────────────────────────────────
# 对外 API / Public API
# ────────────────────────────────────────────────────────────────────
def index_exists() -> bool:
    """
    是否已建索引（至少 collection 存在）。
    Returns True if the target collection is present.

    异常吞掉返回 False：embedded 模式刚启动文件锁未就绪时可能抛错。
    Swallow exceptions; embedded init may briefly fail before locks settle.
    """
    try:
        client = _get_client()
        return settings.rag_collection in {
            c.name for c in client.get_collections().collections
        }
    except Exception:  # noqa: BLE001
        return False


def add_documents_with_sha1(docs: Iterable[Document], sha1: str) -> int:
    """
    批量上插文档，使用 (sha1, chunk_idx) 生成稳定 ID。
    Bulk-upsert with stable IDs derived from (sha1, chunk_idx).

    upsert 是幂等操作：同 ID 重复写会覆盖，不会报错。
    Upsert is idempotent: same ID overwrites; no error on duplicates.

    返回写入的 chunk 数 / Returns the number of chunks written.
    """
    # 转列表（可能传进来是 generator）/ Materialize from possible generator.
    docs = list(docs)
    if not docs:
        return 0
    _ensure_collection()
    client = _get_client()

    # 批量 embed 比逐条快得多（FastEmbed 内部走 batch ONNX）。
    # Batch embed is much faster than per-doc (FastEmbed batches in ONNX).
    texts = [d.page_content for d in docs]
    vectors = _embed(texts)

    # 构造 PointStruct 列表 / Build PointStruct list.
    # PointStruct = (id, vector, payload) 三元组。
    # A point is identified by id, has a vector, and carries arbitrary payload.
    points = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        payload = {
            "page_content": doc.page_content,  # 原文存 payload 便于检索时返回。
            "metadata": doc.metadata,          # 任意元数据 / arbitrary metadata
            "sha1": sha1,                      # ← 用于按文件级删除的 key
            "chunk_idx": i,                    # chunk 在文件内的序号
        }
        points.append(
            qm.PointStruct(
                id=_stable_point_id(sha1, i),
                # tolist() 转 numpy → Python list；Qdrant 不直接吃 ndarray。
                # numpy → Python list (Qdrant doesn't accept ndarray directly).
                vector=vec.tolist(),
                payload=payload,
            )
        )
    # 一次性提交所有 points 到 Qdrant；批量 upsert 性能远好于循环。
    # Submit all points in one batch; far better than looping individual upserts.
    client.upsert(collection_name=settings.rag_collection, points=points)
    return len(points)


def delete_by_sha1(sha1: str) -> None:
    """
    按 sha1 metadata 删除所有 chunks（原地更新的关键路径）。
    Delete all chunks with this sha1 — the key op enabling in-place updates.
    """
    if not index_exists():
        return
    client = _get_client()
    # FilterSelector + must=[FieldCondition(...)] 表达"所有 sha1 == X 的点"。
    # FilterSelector + must=[FieldCondition(...)] selects all points with sha1==X.
    client.delete(
        collection_name=settings.rag_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="sha1",
                        match=qm.MatchValue(value=sha1),
                    )
                ]
            )
        ),
    )


def similarity_search(query: str, k: int = 4) -> list[Document]:
    """
    余弦相似度检索 / Cosine similarity search.
    """
    # 索引不存在直接返回空，避免抛错让上层处理。
    # No index → empty result; let callers decide messaging.
    if not index_exists():
        return []
    client = _get_client()
    # _embed 返回 (1, dim)，取 [0] 拿到单条向量。
    # _embed returns (1, dim); take [0] for the single vector.
    qv = _embed([query])[0]
    hits = client.search(
        collection_name=settings.rag_collection,
        query_vector=qv.tolist(),
        limit=k,
        with_payload=True,   # 让结果带回 payload，否则只有 id + score。
    )
    out: list[Document] = []
    for h in hits:
        # payload 可能为 None（理论上不该，防御性处理）。
        # Defensive guard; payload should always be present but check anyway.
        payload = h.payload or {}
        out.append(
            Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata") or {},
            )
        )
    return out


def drop_index() -> None:
    """
    删除整个 collection（--clear 用）。
    Drop the entire collection (used by --clear).
    """
    client = _get_client()
    if settings.rag_collection in {
        c.name for c in client.get_collections().collections
    }:
        client.delete_collection(settings.rag_collection)


def reset_client() -> None:
    """
    重置客户端（测试或换 persist_dir 时）。
    Reset the client (used by tests or when switching persist_dir).

    embedded 模式下文件锁会持有目录；删除前必须 close()。
    Embedded mode holds a file lock on the dir; must close() before rmtree.
    """
    global _client, _dim
    if _client is not None:
        try:
            _client.close()
        except Exception:  # noqa: BLE001
            # close() 在某些异常状态可能抛；忽略以确保 reset 完成。
            # close() may raise in some states; ignore so reset completes.
            pass
    _client = None
    _dim = None